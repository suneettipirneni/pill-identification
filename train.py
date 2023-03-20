import torch
import os
from torchvision.models import vit_b_32

BASE_DATA_DIR = os.path.join("data", "ePillD_data")

model = vit_b_32()
