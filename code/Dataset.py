import os
import numpy as np
import csv
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Furniture128(Dataset):

    def __init__(self, dir_path, data_type, transform):
        csv_path = os.path.join(dir_path, '{}.csv'.format(data_type))
        self.image_label_df = pd.read_csv(csv_path, dtype={'image_id': int, 'label_id': int})
        self.image_dir = os.path.join(dir_path, data_type)
        self.image_names = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        image_id = name.split('.')[0]
        image_path = os.path.join(self.image_dir, name)
        label = int(self.image_label_df.loc[self.image_label_df['image_id'] == int(image_id)]['label_id']) - 1
        try:
            pil_image = Image.open(image_path).convert('RGB')
            pil_image = self.transform(pil_image)
            return pil_image, label
        except Exception as ex:
            # print(ex)
            return None, label


def collate_fn(seq_list):
    images_list, labels_list = zip(*seq_list)
    batch_images = []
    batch_labels = []
    for i in range(len(images_list)):
        if images_list[i] is not None:
            batch_images.append(images_list[i])
            batch_labels.append(labels_list[i])
    if len(batch_images) > 0:
        return torch.stack(batch_images).to(DEVICE), torch.Tensor(batch_labels).long().to(DEVICE)
    else:
        return None, None

# class Furniture128(Dataset):
#     def __init__(self, preffix: str, transform=None):
#         self.preffix = preffix
        
#         self.labels = {}
#         if preffix != 'test':
#             with open(f'../data/{preffix}.csv', newline='') as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     idx = int(row['image_id'])
#                     id_label = int(row['label_id']) - 1
#                     self.labels[idx] = id_label

#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         idx  = idx + 1
#         p = f'../data/{self.preffix}/{idx}.jpg'
#         try:
#             img = Image.open(p)
#             img = self.transform(img)
#         except:
#             return torch.zeros((3, 224, 224)), 200
#         c, w, h = img.shape
#         if c != 3 or w != 224 or h != 224:
#             return torch.zeros((3, 224, 224)), 200

#         target = self.label[s[idx] if len(self.labels) > 0 else -1

#         return img, target

# def collate_fn(batch):
#     new_img = []
#     new_target = []
#     for img, target in batch:
#         if target == 200:
#             continue
#         c, w, h = img.shape
#         new_img.append(img.view(1, c, w, h))
#         new_target.append(target)
#     if len(new_img) > 0:
#         return torch.cat(new_img).to(DEVICE), torch.from_numpy(np.array(new_target)).to(DEVICE)
#     else:
#         return None, None





