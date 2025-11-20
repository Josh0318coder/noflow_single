import numpy as np
import pandas as pd
from src.utils import (
    CenterPadCrop_numpy,
    Distortion_with_flow_cpu,
    Normalize,
    RGB2Lab,
    ToTensor,
    CenterPad,
    read_flow,
    SquaredPadding
)
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from numpy import random
import os
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
import glob



def image_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy


class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_data_root,
        flow_data_root,
        mask_data_root,
        imagenet_folder,
        annotation_file_path,
        image_size,
        num_refs=5, # max = 20
        image_transform=None,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,  # 保留參數以兼容,但不使用
    ):
        self.video_data_root = video_data_root
        self.flow_data_root = flow_data_root
        self.mask_data_root = mask_data_root
        self.imagenet_folder = imagenet_folder
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.Resize = transforms.Resize(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = transforms.CenterCrop(image_size)
        self.SquaredPadding = SquaredPadding(image_size[0])
        self.num_refs = num_refs

        assert os.path.exists(self.video_data_root), "find no video dataroot"
        assert os.path.exists(self.flow_data_root), "find no flow dataroot"
        assert os.path.exists(self.imagenet_folder), "find no imagenet folder"
        # self.epoch = epoch
        self.image_pairs = pd.read_csv(annotation_file_path, dtype=str)
        self.real_len = len(self.image_pairs)
        # self.image_pairs = pd.concat([self.image_pairs] * self.epoch, ignore_index=True)
        self.real_reference_probability = real_reference_probability
        # nonzero_placeholder_probability 不再使用
        print("##### parsing image pairs in %s: %d pairs #####" % (video_data_root, self.__len__()))

    def __getitem__(self, index):
        (
            video_name,
            prev_frame,
            current_frame,
            flow_forward_name,
            mask_name,
            reference_1_name,
            reference_2_name,
            reference_3_name,
            reference_4_name,
            reference_5_name
        ) = self.image_pairs.iloc[index, :5+self.num_refs].values.tolist()

        video_path = os.path.join(self.video_data_root, video_name)
        # flow_path 和 mask_path 不再需要
        
        current_frame_path = os.path.join(video_path, current_frame)
        list_frame_path = glob.glob(os.path.join(video_path, '*'))
        list_frame_path.sort()
        
        reference_1_path = os.path.join(self.imagenet_folder, reference_1_name)
        reference_2_path = os.path.join(self.imagenet_folder, reference_2_name)
        reference_3_path = os.path.join(self.imagenet_folder, reference_3_name)
        reference_4_path = os.path.join(self.imagenet_folder, reference_4_name)
        reference_5_path = os.path.join(self.imagenet_folder, reference_5_name)
        
        # flow_forward_path 和 mask_path 不再需要
        
        #reference_gt_1_path = prev_frame_path
        #reference_gt_2_path = current_frame_path
        try:
            # 只讀取當前幀,不讀取前一幀
            I2 = Image.open(current_frame_path).convert("RGB")
            
            # 獲取視頻第一幀作為備用參考
            try:
                I_reference_video = Image.open(list_frame_path[0]).convert("RGB") # Get first frame
            except:
                I_reference_video = Image.open(current_frame_path).convert("RGB") # Get current frame if error
            
            reference_list = [reference_1_path, reference_2_path, reference_3_path, reference_4_path, reference_5_path]
            while reference_list: # run until getting the colorized reference
                reference_path = random.choice(reference_list)
                I_reference_video_real = Image.open(reference_path)
                if I_reference_video_real.mode == 'L':
                    reference_list.remove(reference_path)
                else:
                    break
            if not reference_list:
                I_reference_video_real = I_reference_video

            # ✅ 完全移除光流和mask的讀取 - 單幀上色不需要!
            # flow_forward = read_flow(flow_forward_path)  # 不需要
            # mask = Image.open(mask_path)  # 不需要
            
            # transform
            I2 = self.image_transform(I2)
            I_reference_video = self.image_transform(I_reference_video)
            I_reference_video_real = self.image_transform(I_reference_video_real)
            # flow_forward = self.ToTensor(flow_forward)  # 不需要
            # flow_forward = self.Resize(flow_forward)  # 不需要
            

            if np.random.random() < self.real_reference_probability:
                I_reference_output = I_reference_video_real  # Use reference from imagenet
                # ✅ 修改1: 移除 placeholder 生成
                self_ref_flag = torch.zeros_like(I2)  # 改用 I2
            else:
                I_reference_output = I_reference_video  # Use reference from ground truth
                # ✅ 修改2: 移除 placeholder 生成
                self_ref_flag = torch.ones_like(I2)  # 改用 I2

            # ✅ 修改3: outputs 從 10 個元素改為 5 個元素
            # 移除: I1 (前一幀), flow_forward, mask, placeholder
            # 保留: I2 (當前幀), I_reference_output, self_ref_flag, 路徑信息
            outputs = [
                I2,                      # 當前幀
                I_reference_output,      # 參考圖
                self_ref_flag,           # 標記
                video_name + current_frame,  # 當前幀路徑
                reference_path           # 參考圖路徑
            ]

        except Exception as e:
            print("error in reading image pair: %s" % str(self.image_pairs[index]))
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)


def parse_imgnet_images(pairs_file):
    pairs = []
    with open(pairs_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            image_a = line[0]
            image_b = line[1]
            pairs.append((image_a, image_b))
    return pairs


class VideosDataset_ImageNet(data.Dataset):
    def __init__(
        self,
        imagenet_data_root,
        pairs_file,
        image_size,
        transforms_imagenet=None,
        brightnessjitter=0,
        extra_reference_transform=None,
        real_reference_probability=1,
    ):
        self.imagenet_data_root = imagenet_data_root
        self.image_pairs = pd.read_csv(pairs_file, names=['i1', 'i2'])
        self.transforms_imagenet_raw = transforms_imagenet
        self.extra_reference_transform = transforms.Compose(extra_reference_transform)
        self.real_reference_probability = real_reference_probability
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.brightnessjitter = brightnessjitter
        print("##### parsing imageNet pairs in %s: %d pairs #####" % (imagenet_data_root, self.__len__()))

    def __getitem__(self, index):
        pa, pb = self.image_pairs.iloc[index].values.tolist()
        if np.random.random() > 0.5:
            pa, pb = pb, pa

        image_a_path = os.path.join(self.imagenet_data_root, pa)
        image_b_path = os.path.join(self.imagenet_data_root, pb)

        # 单帧模式：直接加载图像，不进行光流扭曲
        I2 = image_loader(image_a_path)
        I_reference_video = image_loader(image_a_path)
        I_reference_video_real = image_loader(image_b_path)

        # 对图像应用变换
        for transform in self.transforms_imagenet_raw:
            I2 = transform(I2)

        # 添加亮度抖动
        I2[0:1, :, :] = I2[0:1, :, :] + torch.randn(1) * self.brightnessjitter

        # 处理参考图像
        I_reference_video = self.extra_reference_transform(I_reference_video)
        for transform in self.transforms_imagenet_raw:
            I_reference_video = transform(I_reference_video)

        I_reference_video_real = self.transforms_imagenet(I_reference_video_real)

        # 选择参考图像
        if np.random.random() < self.real_reference_probability:
            I_reference_output = I_reference_video_real
            self_ref_flag = torch.zeros_like(I2)
        else:
            I_reference_output = I_reference_video
            self_ref_flag = torch.ones_like(I2)

        # 返回单帧数据：当前帧、参考图、标记、路径信息
        return [I2, I_reference_output, self_ref_flag, pb, pa]

    def __len__(self):
        return len(self.image_pairs)



