import requests
import zipfile
import os
from tqdm import tqdm

BASE_DATA_DIR = "./data"
ZIPFILE_NAME = "data.zip"

print("Downloading data...")
response = requests.get("https://github.com/usuyama/ePillID-benchmark/releases/download/ePillID_data_v1.0/ePillID_data.zip", stream=True)

total_size_in_bytes= int(response.headers.get('content-length', 0))
block_size = 1024
progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

if not os.path.exists(BASE_DATA_DIR):
  os.mkdir(BASE_DATA_DIR)

zip_path = os.path.join(BASE_DATA_DIR, "data.zip")

with open(zip_path, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)

progress_bar.close()

print("Extracting download...")

with zipfile.ZipFile(zip_path, "r") as zip:
  zip.extractall("data")

print("Cleaning up...")
os.remove(zip_path)