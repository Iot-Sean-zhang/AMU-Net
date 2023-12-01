from torch.utils import data
import cv2
import os
import numpy as np
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt


# 自定义视频数据集
class VideoDataSet(data.Dataset):

    def __init__(self, root_dir, n_segment=8, mode='train', sample_mod='sparing', modality='RGB', transform=None):

        super(VideoDataSet, self).__init__()
        # 数据集的根路径
        self.root_dir = root_dir
        self.dataset = root_dir.split('/')[-1]
        # train  or validate or test
        self.mode = mode
        self.modality = modality
        self.transform = transform
        # 时间窗口大小
        self.n_segment = n_segment

        # 帧采样模式:dense , parsing , twice
        self.sample_mode = sample_mod

        if self.mode == 'train':
            target_path = os.path.join(self.root_dir, 'train.txt')
        elif self.mode == 'valid':
            target_path = os.path.join(self.root_dir, 'valid.txt')
        elif self.mode == 'test':
            target_path = os.path.join(self.root_dir, 'test.txt')
        else:
            raise NotImplementedError

        with open(target_path, 'r') as f:
            self.video_list = f.readlines()
            self.video_list = [line.strip('\n') for line in self.video_list]
        f.close()

        with open(os.path.join(self.root_dir, 'categories.txt'), 'r') as f:
            self.categories = f.readlines()
            self.categories = [line.strip('\n') for line in self.categories]
        f.close()

        self.totals = self.__len__()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        if self.modality == 'RGB':

            one_path, label, fcount = self.video_list[idx].split(' ')
            label = int(label)
            fcount = int(fcount)
            # 帧采样
            key_frames = self.frame_sampling(fcount, self.sample_mode, n_segment=self.n_segment)
            # print(key_frames)
            # 读视频帧
            video_frames = []
            # 按索引读帧
            for offset in key_frames:
                if self.dataset == 'ActivityNet':
                    template = '{}.jpg'
                elif self.dataset == 'UCF101':
                    template = 'img_{:0>5d}.jpg'
                else:
                    raise NotImplementedError

                img_path = os.path.join(self.root_dir, one_path, template.format(offset))
                video_frames.append(Image.open(img_path).convert('RGB'))

            # 打印有问题视频的路径
            if len(video_frames) == 0:
                print(os.path.join(self.root_dir, one_path))
                raise NotImplementedError

        elif self.modality == 'mp4':

            one_path, label = self.video_list[idx].split(' ')
            label = int(label)

            # 打开视频文件
            f_cap = cv2.VideoCapture(os.path.join(self.root_dir, one_path))
            # w = int(f_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(f_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fcount = int(f_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 帧采样
            key_frames = self.frame_sampling(fcount, self.sample_mode, n_segment=self.n_segment)

            while fcount == 0:
                f_cap.release()
                one_path, label = self.video_list[(idx + random.randint(1, self.__len__())) % self.__len__()].split(' ')
                label = int(label)
                # 打开视频文件
                f_cap = cv2.VideoCapture(os.path.join(self.root_dir, one_path))
                # w = int(f_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # h = int(f_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fcount = int(f_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # 帧采样
                key_frames = self.frame_sampling(fcount, self.sample_mode, n_segment=self.n_segment)

            # 读视频帧
            video_frames = []

            for offset in key_frames:
                # 读取关键帧
                f_cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
                success, frame = f_cap.read()

                if not success:
                    print("no such frame:", offset)
                    # 越界了
                    f_cap.set(cv2.CAP_PROP_POS_FRAMES, offset - 1)
                    success, frame = f_cap.read()

                # 转 RGB
                video_frames.append(Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256))))

            f_cap.release()
            if len(video_frames) == 0:
                # 打印有问题视频的路径
                print(os.path.join(self.root_dir, one_path))
                raise NotImplementedError

            # 帧变形
        video_frames = self.transform(video_frames)
        # label 是 torch.long 类型
        label = torch.from_numpy(np.array(label)).long()

        return video_frames, label

    @staticmethod
    def frame_sampling(nums_frame, strategy, n_segment=3):

        # 稀疏均匀采样
        if strategy == 'sparing':
            duration = nums_frame // n_segment

            if duration > 0:
                offsets = np.array([i * duration for i in range(n_segment)]) + np.random.randint(duration,
                                                                                                 size=n_segment)
            else:
                offsets = [0] * n_segment

        # I3D 密集采样,stride sample:每8个时间步取出一帧,取8次，共64帧
        elif strategy == 'dense':

            sample_pos = max(1, 1 + nums_frame - 64)
            t_stride = 64 // n_segment
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = np.array([(idx * t_stride + start_idx) % nums_frame for idx in range(n_segment)])

        # 双采样
        elif strategy == 'twice':
            tick = nums_frame / float(n_segment)
            if tick > 0:
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(n_segment)] +
                                   [int(tick * x) for x in range(n_segment)])
            else:
                offsets = [0] * n_segment * 2

        # 不采样
        elif strategy == 'no':
            return list(range(nums_frame))
        else:
            raise NotImplementedError
        return offsets
