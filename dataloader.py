import glob

import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import *

img_paths = glob.glob('./img_align_celeba/*.jpg')
num_train, num_val = 200, 20 # demo with small data
train_imgpaths = img_paths[:num_train]
val_imgpaths = img_paths[num_train:num_train + num_val]

class ColorDataset(Dataset):
    def __init__(self, img_paths, data_len=2880, image_size=(128, 128)):
        if data_len > 0:
            self.img_paths = img_paths[:int(data_len)]
        else:
            self.img_paths = img_paths
        self.tfs = transforms.Resize((image_size[0], image_size[1]))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        arr_img_bgr = cv2.imread(img_path)
        arr_img_lab = cv2.cvtColor(arr_img_bgr, cv2.COLOR_BGR2LAB)
        arr_img_lab = ((arr_img_lab * 2.0) / 255.0) - 1.0
        tens_img_lab = torch.tensor(arr_img_lab.transpose(2, 0, 1), dtype=torch.float32)
        original_img_l = tens_img_lab[:1, :, :]
        tens_img_lab = self.tfs(tens_img_lab)
        tens_img_l = tens_img_lab[:1, :, :]
        tens_img_ab = tens_img_lab[1:, :, :]
        return original_img_l, tens_img_l, tens_img_ab

    def __len__(self):
        return len(self.img_paths)
    
train_dataset = ColorDataset(train_imgpaths, num_train)
val_dataset = ColorDataset(val_imgpaths, num_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)