import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
import os

class AmazonProductDataset(Dataset):
    """
    Loads the Amazon products CSV, downloads images,
    and returns dicts with 'title', 'desc', 'image', 'asin', plus the full row.
    I use data from here https://github.com/luminati-io/eCommerce-dataset-samples/blob/main/amazon-products.csv
    """
    def __init__(self, csv_path, image_dir="images", max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(max_samples).reset_index(drop=True)
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = str(row['title']) if pd.notna(row['title']) else ""
        desc = str(row['description']) if pd.notna(row['description']) else ""
        image_url = row.get('image_url', None)
        img = self.load_image(image_url, row['asin'])
        return {
            'title': title,
            'desc': desc,
            'image': img,
            'asin': row['asin'],
            'row': row.to_dict()
        }
    
    def load_image(self, url, asin):
        if not url or pd.isna(url):
            return None
        path = os.path.join(self.image_dir, f"{asin}.jpg")
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content)).convert('RGB')
                    img.save(path)
            except Exception:
                return None
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            return None