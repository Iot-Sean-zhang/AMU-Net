from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math

DefaultActivation = nn.LeakyReLU(negative_slope=0.05, inplace=True)


def _load_state_dict(model, path):
    import re
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
    )
    state_dict = torch.load(path)

    for key in list(state_dict.keys()):

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def replace_relu(net, act=nn.Identity()):
    for name, module in net.named_modules():
        if hasattr(module, 'relu'):
            module.relu = act
    return net


def normed_laplace(A):
    """
    :param A:  带权图的邻接矩阵
    :return: L, 拉普拉斯矩阵
    """
    # 单位矩阵
    # E = torch.eye(A.shape[1], device=A.device).repeat(A.shape[0], 1, 1)
    # A = A + E
    # 带权度矩阵
    degree = A.sum(dim=-1)
    degree = torch.clamp(degree, min=1e-8)
    D = []
    for d in degree:
        # 归一化的度矩阵
        D.append(torch.diag(d.pow(-0.5)))

    D = torch.cat(D, dim=0).view(-1, A.shape[1], A.shape[2])

    L = D @ A @ D

    return L


class AccurateMotionPerception(nn.Module):

    def __init__(self, chs, t, reduction=8):
        """
         :param chs: input channels
         :param t: frames
         :param reduction: reduction factor
        """
        super(AccurateMotionPerception, self).__init__()
        self.T = t  # number of frames
        r = chs // reduction  # reduce channels
        self.align = nn.Conv2d(chs, chs, kernel_size=3, stride=1, padding=1, groups=chs)  # 3*3 group conv
        self.conv_5 = nn.Conv2d(2, 1, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)  # 5*5 conv
        # depth-wise convolution
        self.agg = nn.Sequential(
            nn.Conv2d(2 * chs, chs, groups=chs, kernel_size=3, stride=1, padding=1, bias=False),  # 2*chs -> chs
            nn.BatchNorm2d(chs),
            nn.Conv2d(chs, r, kernel_size=1, bias=False),  # chs ->r
            nn.BatchNorm2d(r),
            DefaultActivation,
            nn.Conv2d(r, chs, kernel_size=1, bias=False),  # r->chs
            nn.BatchNorm2d(chs),
        )

    def shuffle_channel(self, x, num_groups):
        """
            mean shuffle
        """
        batch_size, num_channels, height, width = x.size()  # shape
        assert num_channels % num_groups == 0
        x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # swap different groups
        return x.contiguous().view(batch_size, num_channels, height, width)

    def forward(self, X):
        NT, C, H, W = X.shape  # tensor shape
        # ---------bi-direction feature differences------------------------------
        X_align = self.align(X).view((-1, self.T, C, H, W))  # for geometric deformation
        X = X.view((-1, self.T, C, H, W))  # N, T, C, H

        X_b1, b1 = torch.split(X, [self.T - 1, 1], dim=1)  # [0,T-1],[T]
        X_f2, f2 = torch.split(X_align, [self.T - 1, 1], dim=1)  # [0,T-1],[T]
        f1, X_f1 = torch.split(X_align, [1, self.T - 1], dim=1)  # [0],[1,T]
        b2, X_b2 = torch.split(X, [1, self.T - 1], dim=1)  # [0],[1,T]
        X_diff_f = X_f1 - X_b1  # forward difference
        X_diff_b = X_f2 - X_b2  # backward difference

        X_motion1 = torch.cat([f1 - b1, X_diff_f], dim=1)  # forward motion
        X_motion1 = X_motion1.view((-1, C, H, W))  # NT,C,H,W
        X_motion2 = torch.cat([f2 - b2, X_diff_b], dim=1)  # backward motion
        X_motion2 = X_motion2.view((-1, C, H, W))  # NT,C,H,W
        X_motion = X_motion1
        X_motion = torch.cat((X_motion1, X_motion2), dim=1)  # NT,2C,H,W
        X_motion = self.shuffle_channel(X_motion, C)  # channel shuffle
        X_motion = self.agg(X_motion)  # NT,C,H,W, fused motion maps

        # ----------------motion space attention-----------------------------
        avg_out1 = torch.mean(X_motion, dim=1, keepdim=True)  # spatial avg pool
        max_out1, _ = torch.max(X_motion, dim=1, keepdim=True)  # spatial max pool
        query = torch.cat((avg_out1, max_out1), dim=1)  # query
        scores_s = self.conv_5(query)  # attention  scores
        X_motion = torch.sigmoid(scores_s) * X_motion  # attention

        # ---------------------feature excitation----------------------------
        X = X.view((-1, C, H, W))  # NT,C,H,W
        out = X * torch.tanh(X_motion) + X  # excitation + residual connection

        return out


class MotionedDenseBlock(nn.Module):

    def __init__(self, net, chs, num_frames=8, r=8):
        super(MotionedDenseBlock, self).__init__()
        self.net = net
        self.chs = chs
        self.ma = AccurateMotionPerception(chs, num_frames, reduction=r)

    def forward(self, X):
        X = self.net(X)
        X = self.ma(X)

        return X


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))

        # 初始化参数
        self.init_parameters()

    def init_parameters(self):

        nn.init.kaiming_uniform_(self.lin.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, L, X):

        X = self.lin(X)
        output = L @ X

        if self.use_bias:
            output += self.bias

        return output


class GCN(nn.Module):

    def __init__(self, input_dims, out_dims, n_layers=3):

        super(GCN, self).__init__()
        self.gcn = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers - 1:
                self.gcn.append(GraphConvolution(input_dims, out_dims, use_bias=False))
            else:
                self.gcn.append(GraphConvolution(input_dims, input_dims, use_bias=False))
        self.att_gcn = GraphConvolution(out_dims, 1)
        self.act = DefaultActivation

    def forward(self, L, X):

        res = X
        for m in self.gcn:
            X = self.act(m(L, X))  # b, n, c
        att_scores = torch.softmax(self.att_gcn(L, X + res), dim=1)  # b,n,1
        read_out = (X * att_scores).sum(dim=1)

        return read_out


class ActionInstanceGraphModule(nn.Module):

    def __init__(self, dims, t=8, num_pos=3, n_layers=2, reduction=8):
        super(ActionInstanceGraphModule, self).__init__()
        self.t = t  # number of frames
        self.num_pos = num_pos  # number of motion objects
        self.dims = dims  # input channels

        self.detector = nn.Sequential(
            # squeeze
            nn.Conv3d(dims, dims // reduction, kernel_size=1),
            nn.BatchNorm3d(dims // reduction),
            DefaultActivation,
            # space-time saliency detection
            nn.Conv3d(dims // reduction, dims // reduction, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1),
            nn.Conv3d(dims // reduction, num_pos, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1),
            nn.BatchNorm3d(num_pos),
            nn.Sigmoid(),
        )
        # N graph convolution layers followed by a weighted aggregation layer
        self.gcn = GCN(dims, dims, n_layers=n_layers)

    def forward(self, X):
        nt, c, h, w = X.shape  # input shape
        n = nt // self.t  # batch_size
        X = X.view(n, self.t, c, h, w)  # n,t,c,h,w
        residual = X.mean([3, 4])  # spatial avg pool
        X = X.permute(0, 2, 1, 3, 4)  # n, c ,t ,h, w
        s_map = self.detector(X)  # n, num_pos, t,  h , w
        # print(s_map.shape)
        # n, c, 1 ,t, h, w  * n, 1, num_pos, t, h, w -> n,  c, num_pos, t, h, w
        nodes = (X.unsqueeze(dim=2) * s_map.unsqueeze(dim=1)) \
            .mean([4, 5]).permute(0, 3, 2, 1).contiguous().view(n, -1, c)  # n, t * num_pos , c

        e = nodes @ nodes.permute(0, 2, 1) / c  # Adjacency matrix
        laplace = normed_laplace(e)  # D^-1 @ A @ D^-1

        gcn_out = self.gcn(laplace, nodes)  # graph reasoning and aggregation
        out = torch.cat([gcn_out.unsqueeze(dim=1), residual], dim=1)  # n, t+1, c

        return out


class MotionPercievedDenseNet(nn.Module):

    def __init__(self, num_classes, num_frames, drop=0.5, pretrained=False, fusion='avg',
                 pretrain_path=None, base_model='resNet50'):

        super(MotionPercievedDenseNet, self).__init__()
        self.T = num_frames

        if base_model == 'densenet121':

            if pretrained and pretrain_path:
                back = torchvision.models.densenet121(pretrained=False)
                _load_state_dict(back, pretrain_path)
            elif pretrained and pretrain_path is None:
                back = torchvision.models.densenet121(pretrained=True)
            else:
                back = torchvision.models.densenet121(pretrained=False)

            self.add_ma2dense(back, num_frames)
            self.features = back.features
            cnn_feature_dim = back.classifier.in_features

        elif base_model == 'resNet50':

            if pretrained and pretrain_path:
                back = torchvision.models.resnet50(pretrained=False)
                back.load_state_dict(torch.load(pretrain_path))

            elif pretrained and pretrain_path is None:
                back = torchvision.models.resnet50(pretrained=pretrained)

            else:
                back = torchvision.models.resnet50(pretrained=False)

            back = replace_relu(back, act=DefaultActivation)
            self.add_ma2Res(back, n_segment=num_frames)
            self.features = nn.Sequential(*list(back.children())[:-2])
            cnn_feature_dim = list(back.children())[-1].in_features

        elif base_model == 'mobileNetV2':
            back = torchvision.models.mobilenet_v2(pretrained=False)
            back.load_state_dict(torch.load('./pretrained/mobilenet_v2-b0353104.pth'))
            self.add_ma2Mv2(back, n_segments=num_frames)
            self.features = back.features
            cnn_feature_dim = 1280
        elif base_model == 'shuffleNetV2':
            back = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
            back.load_state_dict(torch.load('./pretrained/shufflenetv2_x1-5666bf0f80.pth'))
            self.add_ma2Sv2(back, num_frames)
            self.features = nn.Sequential(
                *list(back.children())[:-1]
            )
            cnn_feature_dim = 1024

        elif base_model == 'squeezeNet':

            back = torchvision.models.squeezenet1_1(pretrained=False)
            back.load_state_dict(torch.load('./pretrained/squeezenet1_1-b8a52dc0.pth'))
            self.add_ma2Squeeze(back.features, num_frames)
            self.features = back.features
            cnn_feature_dim = 512

        else:
            raise NotImplementedError

        self.fusion = fusion
        # self.feat_maps = None
        if fusion == 'lat':
            self.space_time_fuse = ActionInstanceGraphModule(cnn_feature_dim, num_frames, num_pos=3, n_layers=2)
        else:
            self.final_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(cnn_feature_dim, num_classes),
        )

    def forward(self, X):

        out = self.features(X)

        if self.fusion == 'avg':

            out = self.final_pool(out)
            out = self.classifier(out)
            _, C = out.shape
            out = torch.mean(out.view(-1, self.T, C), dim=1)

        elif self.fusion == 'lat':

            out = self.space_time_fuse(out)
            out = self.classifier(out)
            out = torch.mean(out, dim=1)

        else:
            pass

        return out

    def add_ma2Sv2(self, model, n_segment=8):

        def make_block_temporal(stage, in_planes, n_segment):
            n_round = 1
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].branch2[0] = MotionedDenseBlock(b.branch2[0], chs=in_planes, num_frames=n_segment, r=4)

            return nn.Sequential(*blocks)

        model.stage2 = make_block_temporal(model.stage2, 58, n_segment)
        model.stage3 = make_block_temporal(model.stage3, 116, n_segment)
        model.stage4 = make_block_temporal(model.stage4, 232, n_segment)

    def add_ma2Squeeze(self, model, n_segments=8):

        chs_list = [16, 16, 32, 32, 48, 48, 64, 64]
        idxs = [3, 4, 6, 7, 9, 10, 11, 12]

        model[2] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        model[5] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        model[8] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i, id in enumerate(idxs):
            model[id].squeeze = MotionedDenseBlock(model[id].squeeze, chs=chs_list[i], num_frames=n_segments)

    def add_ma2Mv2(self, model, n_segments=8):

        chs_list = [32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960]
        n_round = 1
        for i, cbr in enumerate(model.features[1:18]):

            if chs_list[i] >= 384:
                n_round = 3
            if i % n_round == 0:
                cbr.conv[0] = MotionedDenseBlock(cbr.conv[0], chs=chs_list[i], num_frames=n_segments)

    def add_ma2dense(self, backbone, T, ch_list=[64, 128, 256, 512], increase=32, n_round=1):

        def make_block(stage, T, start_chs, n_round):
            blocks = list(stage.children())
            if len(blocks) > 23:
                n_round = 2
            else:
                n_round = 1
            for i, b in enumerate(blocks):
                chs = start_chs + increase * i
                if i % n_round == 0:
                    blocks[i].conv1 = MotionedDenseBlock(b.conv1, chs, T)

        make_block(backbone.features.denseblock1, T, ch_list[0], n_round)
        make_block(backbone.features.denseblock2, T, ch_list[1], n_round)
        make_block(backbone.features.denseblock3, T, ch_list[2], n_round)
        make_block(backbone.features.denseblock4, T, ch_list[3], n_round)

    def add_ma2Res(self, net, ch_list=[64, 128, 256, 512], n_segment=8):

        n_round = 1

        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage, in_planes, n_segment):

            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = MotionedDenseBlock(b.conv1, chs=in_planes, num_frames=n_segment, r=8)

            return nn.Sequential(*blocks)

        # pdb.set_trace()
        net.layer1 = make_block_temporal(net.layer1, ch_list[0], n_segment)
        net.layer2 = make_block_temporal(net.layer2, ch_list[1], n_segment)
        net.layer3 = make_block_temporal(net.layer3, ch_list[2], n_segment)
        net.layer4 = make_block_temporal(net.layer4, ch_list[3], n_segment)


if __name__ == '__main__':

    x = torch.randn((8, 3, 224, 224))
    net = MotionPercievedDenseNet(num_frames=8, num_classes=200,
                                  drop=0.5, pretrained=True,
                                  base_model='resNet50',
                                  pretrain_path=None, fusion='lat')
    # from TEA.master.mian import get_tea50
    # net = get_tea50()
    # from TDN.master.model import tdn_net
    # net = tdn_net("resNet50", 8, False, 200)
    # from TSM.master.model import TSMNet
    #
    # net = TSMNet(nums_class=200, n_segment=8, n_div=16,
    #              dropout=0.5, back="mobileNetV2", pretrained=False)
    # print(net)
    
    # from ActionNet.master.model import ActionNet
    # net = ActionNet(back="resNet50", nums_class=200, n_segment=8,
    #                 dropout=0.5, pretrained=True, pretrain_path=None)
    # from GSM.master.main import get_gsm
    # net = get_gsm()
    # from TimeSformer.timesformer.models.vit import TimeSformer
    #
    # net = TimeSformer(img_size=224, patch_size=16, num_classes=200, num_frames=8, attention_type='divided_space_time')
    device = torch.device('cuda:0')
    print(net)
    net.to(device)
    x = x.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(100):
        with torch.no_grad():
            _ = net(x)

    iterations = 10
    times = torch.zeros(iterations)
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = net(x)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    #
    # net = MotionPercievedDenseNet(
    #     num_frames=8, num_classes=200,
    #     drop=0.5, pretrained=True,
    #     base_model='resNet50',
    #     pretrain_path=None, fusion='avg'
    # )
    #
    # x = (torch.rand(size=(8, 3, 224, 224), dtype=torch.float32),)
    # flops = FlopCountAnalysis(net, x)
    # print("FLOPs: {:.2e}".format(flops.total()))
    # print(parameter_count_table(net))
