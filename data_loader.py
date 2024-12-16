# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : Haonan Wang
# @File    : Load_Dataset.py
# @Software: PyCharm
import PIL
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class BasicDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_list = os.listdir(self.dataset_path)
        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.ToTensor(),
                                             normalize])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image_path = os.path.join(self.dataset_path, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"图像 {image_filename} 读取失败。")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        transformed_image = self.transform(image)

        # 保存预处理后的图像（仅用于调试，避免生产环境中使用）
        # if idx == 0:
        #     transforms.ToPILImage()(transformed_image).save("transformed_image_debug.jpg")

        return transformed_image, image_filename

    # def __getitem__(self, idx):
    #     image_filename = self.images_list[idx]
    #     image = cv2.imread(os.path.join(self.dataset_path, image_filename))
    #     image = PIL.Image.fromarray(image)
    #
    #     return self.transform(image), image_filename



