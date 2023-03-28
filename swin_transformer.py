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
from datasets import SingleImgPillID, SiamesePillID, BalancedBatchSamplerPillID
from torch.utils.tensorboard import SummaryWriter
from train_test import train, test
import pickle


def run_main(args):

    FOLDS_DIR = os.path.join(args.data_root_dir,'folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base') 
    NUM_CLASSES = 9804

    args.label_encoder = os.path.join(args.data_root_dir, args.label_encoder)
    args.all_imgs_csv = os.path.join(args.data_root_dir, args.all_imgs_csv)

    csv_files = glob(os.path.join(FOLDS_DIR,'*.csv'))
    all_imgs_df = [x for x in csv_files if x.endswith('all.csv')]
    train_files = sorted([x for x in csv_files if not x.endswith('all.csv')])
    test_file = train_files.pop(-1)



    train_df = []
    for i in range(len(train_files)):
        train_df.append(pd.read_csv(train_files[i]))

    test_df = pd.read_csv(test_file)
    all_imgs_df = pd.read_csv(args.all_imgs_csv)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_imgs_df['label'])

    pickle.dump(label_encoder, open(args.label_encoder, "wb"))



    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



    train_dataset = [ SingleImgPillID(df=x, label_encoder=label_encoder, transform=transform, train=True) for x in train_df ]
    test_dataset = SingleImgPillID(df=test_df, label_encoder=label_encoder, transform=transform, train=False)

    train_loaders = [ DataLoader(x, batch_size=args.batch_size, num_workers=args.num_workers) for x in train_dataset ]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = swin_v2_t(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=5e-5)

    writer = SummaryWriter()

    print('Training Model')
    for epoch in range(args.num_epochs):
        train(model, optimizer, train_loaders, device, args.batch_size, epoch, writer)
        print()
        if epoch % 10 == 0 and epoch != 0:
            print('===================Test===================')
            test(model, test_loader, device, args.batch_size, 1, writer)
            print('==========================================')
            print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Swin Transformer')

    parser.add_argument('--batch_size', type=int, default=32, help='divide data_loader by batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of parallel processes in dataloader')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs to train')
    
    parser.add_argument('--data_root_dir', default="data/ePillID_data")
    parser.add_argument('--all_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    parser.add_argument('--val_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    parser.add_argument('--test_imgs_csv', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    parser.add_argument('--label_encoder', default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle")

    args = None

    args, unparsed = parser.parse_known_args()

    run_main(args)

