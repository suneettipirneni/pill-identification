import os
import argparse
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import swin_v2_t
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import pandas as pd
from datasets import SingleImgPillID, SiamesePillID
from torch.utils.tensorboard import SummaryWriter
from train_test import train, test


class SwinTransformer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = swin_v2_t()
        self.model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    def forward(self,x):
        return self.model(x)



def run_main(FLAGS):
    BASE_DATA_DIR = os.path.join("data", "ePillID_data")
    SPLITS_DIR = os.path.join(BASE_DATA_DIR,'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base') 
    NUM_CLASSES = 4902


    train_encoder = LabelEncoder()
    test_encoder = LabelEncoder()

    csv_files = glob(os.path.join(SPLITS_DIR,'*.csv'))
    all_imgs_df = [x for x in csv_files if x.endswith('all.csv')]
    train_files = sorted([x for x in csv_files if not x.endswith('all.csv')])
    test_file = train_files.pop(-1)


    train_df = []
    for i in range(len(train_files)):
        train_df.append(pd.read_csv(train_files[i]))

    train_encoder.fit(train_df[0]['label'])

    test_df = pd.read_csv(test_file)
    test_encoder.fit(test_df['label'])


    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



    train_dataset = SingleImgPillID(df=train_df[0], label_encoder=train_encoder, transform=transform, train=True)
    test_dataset = SingleImgPillID(df=test_df, label_encoder=test_encoder, transform=transform, train=False)


    # train_dataset = SiamesePillID(df=test_df, transform=test_transform, train=False)
    # test_dataset = SiamesePillID(df=test_df, transform=test_transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer(NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=1e-3)

    writer = SummaryWriter()

    print('Training Model')
    for epoch in range(FLAGS.num_epochs):
        train(model, optimizer, train_loader, device, FLAGS.batch_size, epoch, writer)
        print()
        if epoch % 25 == 0 and epoch is not 0:
            print('===================Test===================')
            test(model, optimizer, test_loader, device, FLAGS.batch_size, 1, writer)
            print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Swin Transformer')

    parser.add_argument('--batch_size', type=int, default=32, help='divide data_loader by batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of parallel processes in dataloader')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train')

    FLAGS = None

    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)

