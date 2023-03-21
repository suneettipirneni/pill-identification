import torch
import os
from torchvision.models import vit_b_32
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_loaders import SingleImgPillID

label_encoder = LabelEncoder()

BASE_DATA_DIR = os.path.join("data", "ePillID_data")

images_df = pd.read_csv(os.path.join(BASE_DATA_DIR, "all_labels.csv"))

label_encoder.fit(images_df['label'])

augmentations = transforms.Compose([])

train_loader = SingleImgPillID(df=images_df, label_encoder=label_encoder, train=True)

# print(images_df.head())
model = vit_b_32()

for item in train_loader:
  image = item['image']
  label = item['label']
  