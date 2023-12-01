import torch
from torch.nn.utils import clip_grad_norm_
import time
import os
import tqdm
# 评价指标类
from torch.optim.lr_scheduler import MultiStepLR


class Indicator(object):
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def accumulate(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.shape[0]
    # 每个样本预测概率最大的前K个对应的类别
    _, pred = output.topk(max_k, 1, True, True)
    # 转置
    pred = pred.t()
    # 标签竖向复制后与预测值比较
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # 最有可能的前k个预测值在标签中命中数 / 总样本数
    for k in top_k:
        # 统计True的个数
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # 计算 acc top-K，(%)
        res.append(correct_k * 100.0 / batch_size)

    return res


def train_model(net, train_iter, valid_iter, epochs=100, lr=0.01, momentum=0.9, weight_decay=1e-4, top_k=(1, 5),
                milestones=[20, 30, 40], manner='begin', check_path=None, device=None, clip_norm=-1, eval_freq=5,
                save_start=20, log_file=None, save_path=None):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

    # 检查点恢复
    if manner == 'resume':

        checkpoint = torch.load(check_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        print('---------- resume to train model at epoch %d--------------' % start_epoch)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # optimizer的所有参数放在GPU上
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device[0])

    else:
        start_epoch = 0
        if os.path.exists(log_file):
            os.remove(log_file)

    net = torch.nn.DataParallel(net, device_ids=device)
    net = net.to(device[0])

    best_acc = 0.0

    for e in range(start_epoch, epochs):
        net.train()
        # 记录损失
        loss = Indicator()
        # 记录acc1
        top1 = Indicator()
        # 记录acc5
        top5 = Indicator()

        # 记录迭代次数
        count = 0
        # 进度条
        pbar = tqdm.tqdm(
            total=len(train_iter),
            desc=f"Epoch {e + 1}/{epochs}",
            postfix=dict,
            miniters=0.3,
            ncols=120
        )
        # 训练核心代码
        for X, Y in train_iter:
            # 记录一个批量数据加载所需时间
            count += 1
            # print(X.shape)
            N, T, C, H, W = X.shape
            X = X.view((-1, C, H, W))
            X, Y = X.to(device[0]), Y.to(device[0])

            optimizer.zero_grad()
            pred = net(X)
            err = criterion(pred, Y)
            loss.accumulate(err.item(), N)
            acc1, acc5 = accuracy(pred, Y, top_k=top_k)
            top1.accumulate(acc1.item(), N)
            top5.accumulate(acc5.item(), N)
            err.backward()  # 计算梯度后释放计算图
            # 梯度裁剪
            if clip_norm != -1:
                clip_grad_norm_(net.parameters(), clip_norm)
            # SGD
            optimizer.step()
            # 进度条显示
            pbar.set_postfix(
                **{'loss': loss.avg, 'lr': optimizer.state_dict()['param_groups'][0]['lr'], 'acc@1': top1.avg,
                   'acc@5': top5.avg})
            pbar.update(1)
        pbar.close()
        # 学习率调整
        scheduler.step()
        # 隔几轮评估以下
        if (e + 1) % eval_freq == 0:
            # 模型评估
            valid_time, valid_acc1, valid_acc5 = validate(net, valid_iter, top_k=top_k, device=device)

            if e >= save_start and valid_acc1 > best_acc:
                # 在30轮以后，只保存评估acc1最好的参数
                torch.save(net.module.state_dict(), save_path)
                best_acc = valid_acc1
                print('----------best acc {best_acc:.4f}, model has been saved-----------'.format(best_acc=best_acc))

            out_info = ('Epoch:{0}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Train Acc@1  {top1.avg:.3f}\t'
                        'Train Acc@5  {top5.avg:.3f} \t'
                        'Valid Time   {valid_time:.3f}\t'
                        'Valid Acc@1  {valid_acc1:.3f}\t'
                        'Valid Acc@5  {valid_acc5:.3f}\t'.format(e + 1, loss=loss, top1=top1, top5=top5,
                                                                 valid_time=valid_time,
                                                                 valid_acc1=valid_acc1, valid_acc5=valid_acc5)
                        )

            # 记录训练信息到log.txt
            with open(log_file, 'a') as f:
                f.write(out_info)
                f.write('\n')
            f.close()

            print(out_info)

        # 保存断点
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, check_path)


def validate(net, valid_iter, top_k=(1, 5), device=None):
    # 记录acc1
    top1 = Indicator()
    # 记录acc5
    top5 = Indicator()
    # 记录训练时间
    batch_time = Indicator()
    # 开启评估模式
    net.eval()
    with torch.no_grad():
        # 训练核心代码
        for X, Y in valid_iter:
            end = time.time()
            N, T, C, H, W = X.shape
            X = X.view((-1, C, H, W))
            X, Y = X.to(device[0]), Y.to(device[0])
            pred = net(X)
            acc1, acc5 = accuracy(pred, Y, top_k=top_k)
            top1.accumulate(acc1.item(), N)
            top5.accumulate(acc5.item(), N)
            # 记录一个批量的时间
            batch_time.accumulate(time.time() - end)

    return batch_time.avg, top1.avg, top5.avg


# 保存断点以恢复执行
def save_checkpoint(state, filename):
    torch.save(state, filename)
