import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        except:
            return None, label
            

def collate_fn(seq_list):
    images_list, labels_list = zip(*seq_list)
    batch_images = []
    batch_labels = []
    for i in range(len(images_list)):
        if images_list[i] is not None:
            batch_images.append(images_list[i])
            batch_labels.append(labels_list[i])
    return torch.stack(batch_images).to(DEVICE), torch.Tensor(batch_labels).long().to(DEVICE)





