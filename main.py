
from transforms import *
from dataset import *
from config import *
from torch.utils.data import DataLoader
from train import train_model
from model import MotionPercievedDenseNet
import os


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return None


if __name__ == '__main__':

    data_cg = data_config
    mode_cg = model_config
    train_cg = train_config

    train_trans = torchvision.transforms.Compose([
        # GroupScale(256),
        GroupRandomCrop(224),
        GroupRandomHorizontalFlip(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )
    ])
    val_trans = torchvision.transforms.Compose([
        # GroupScale(256),
        GroupCenterCrop(224),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )
    ])

    train_data = VideoDataSet(root_dir=os.path.join(data_cg['root_dir'], data_cg['dataset']),
                              n_segment=data_cg['n_segment'], mode='train', modality=data_cg['modality'],
                              sample_mod=data_cg['sample_mod'], transform=train_trans)

    valid_data = VideoDataSet(root_dir=os.path.join(data_cg['root_dir'], data_cg['dataset']),
                              n_segment=data_cg['n_segment'], mode='valid', modality=data_cg['modality'],
                              sample_mod=data_cg['sample_mod'], transform=val_trans)

    train_iter = DataLoader(train_data, batch_size=data_cg['batch_size'], shuffle=True,
                            num_workers=data_cg['num_workers'], pin_memory=data_cg['pin_mem'])
    valid_iter = DataLoader(valid_data, batch_size=data_cg['batch_size'], shuffle=True,
                            num_workers=data_cg['num_workers'], pin_memory=data_cg['pin_mem'])

    print("""
     DataSet Configs are as Follows:
          dataset:{}
          modality:{}
          sampling:{}
          n_segment:{}
          batch_size:{}
    """.format(data_cg['dataset'], data_cg['modality'], data_cg['sample_mod'], data_cg['n_segment'],
               data_cg['batch_size']))

    net = MotionPercievedDenseNet(num_frames=mode_cg['num_frames'], num_classes=mode_cg['num_classes'],
                                  drop=mode_cg['drop'], pretrained=mode_cg['pretrained'],
                                  base_model=mode_cg['base_model'],
                                  pretrain_path=mode_cg['pretrain_path'], fusion=mode_cg['fusion'])
    print(net)
    print(("""
     Model Configs are as Follows:
          num_frames:{}
          classes:{}
          dropout:{}
          pretrain:{}
          base:{}
          pretrain_path:{}
          fusion:{}
    """.format(mode_cg['num_frames'], mode_cg['num_classes'], mode_cg['drop'], mode_cg['pretrained'],
               mode_cg['base_model'], mode_cg['pretrain_path'], mode_cg['fusion'])))

    print("""
     Traing Configs are as Follows:
          epochs:{}
          lr:{}
          train_manner:{}
          milestones:{}
          grad_clip:{}
          eval_freq:{}
          check_path:{}
          weight_decay:{}
    """.format(train_cg['epochs'], train_cg['lr'], train_cg['manner'], train_cg['milestones'],
               train_cg['grad_clip'], train_cg['eval_freq'], train_cg['check_path'], train_cg['weight_decay']))

    devices = [i for i in range(train_cg['gpus']) if try_gpu(i) is not None]

    print('-----gpus: ', devices)

    print('---------------------train start---------------------------')

    train_model(net, train_iter=train_iter, valid_iter=valid_iter, epochs=train_cg['epochs'], lr=train_cg['lr'],
                momentum=train_cg['momentum'], weight_decay=train_cg['weight_decay'], top_k=train_cg['top_k'],
                milestones=train_cg['milestones'], manner=train_cg['manner'], check_path=train_cg['check_path'],
                device=devices, clip_norm=train_cg['grad_clip'],
                eval_freq=train_cg['eval_freq'], log_file=train_cg['log_file'], save_path=train_cg['save_path'])

    print('---------------------train end-----------------------------')

