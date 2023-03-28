import sys
from pathlib import Path
sys.path.append(str(Path().parent))

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
import src.arguments as arguments
import src.train_nocv as train_nocv

NUM_CLASSES = 9804
args = arguments.cv_parser().parse_args()

# TODO: fix path issue when data_root_dir is not aboslute path
# assert os.path.isabs(args.data_root_dir)
# find csv files and prepare args for each run
args.folds_csv_dir = os.path.join(args.data_root_dir, args.folds_csv_dir)
args.label_encoder = os.path.join(args.folds_csv_dir, "label_encoder.pickle")

# make sure csv_files are sorted
csv_files = glob(os.path.join(args.folds_csv_dir, "*.csv"))
args.all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]

csv_files = sorted([x for x in csv_files if not x.endswith("all.csv")])

args.test_imgs_csv = csv_files.pop(-1)  # use the last fold as hold out

print("val csv files: ",csv_files)

metrics_dfs_list = []
predictions_dfs_list = []
for i, val_csv in enumerate(csv_files):
    args.val_imgs_csv = val_csv

    metrics_df, predictions_df = train_nocv.run(args)
    metrics_dfs_list.append(metrics_df)
    predictions_dfs_list.append(predictions_df)

all_metrics_df = pd.concat(metrics_dfs_list, ignore_index = True)





# transform = transforms.Compose([
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])



# train_dataset = [ SingleImgPillID(df=x, label_encoder=label_encoder, transform=transform, train=True) for x in train_df ]
# test_dataset = SingleImgPillID(df=test_df, label_encoder=label_encoder, transform=transform, train=False)

# train_loaders = [ DataLoader(x, batch_size=args.batch_size, num_workers=args.num_workers) for x in train_dataset ]
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = swin_v2_t(num_classes=NUM_CLASSES, dropout=0.5).to(device)
# optimizer = optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=5e-5)

# writer = SummaryWriter()

# print('Training Model')
# for epoch in range(args.num_epochs):
#     train(model, optimizer, train_loaders, device, args.batch_size, epoch, writer)
#     print()
#     if epoch % 10 == 0 and epoch != 0:
#         print('===================Test===================')
#         test(model, test_loader, device, args.batch_size, 1, writer)
#         print('==========================================')
#         print()


