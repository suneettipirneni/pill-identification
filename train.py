import torch
import os
from torchvision.models import vit_b_32
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

BASE_DATA_DIR = os.path.join("data", "ePillID_data")
NUM_EPOCHS = 25
NUM_CLASSES = 4902
BATCH_SIZE = 5

from datasets import SingleImgPillID

label_encoder = LabelEncoder()



images_df = pd.read_csv(os.path.join(BASE_DATA_DIR, "all_labels.csv"))

label_encoder.fit(images_df['label'])

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((224, 224), antialias=True)
])

train_dataset = SingleImgPillID(df=images_df, label_encoder=label_encoder, transform=transform, train=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vit_b_32(num_classes=NUM_CLASSES)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model.to(device)

for epoch in range(NUM_EPOCHS):
  running_loss = 0.0
  for item in tqdm(train_loader):
    image = item['image'].to(device)
    label = item['label'].to(device)
    image_name = item['image_name']
    is_ref = item['is_ref']

    optimizer.zero_grad()
    outputs: torch.Tensor = model(image)
    
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f"epoch {epoch} loss = {running_loss}")





  