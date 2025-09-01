import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class IQADataset(Dataset):
    """
    Dataset class for Image Quality Assessment datasets.
    Currently supports KonIQ-10k and LIVE-IW.
    
    Args:
        root (str): Root directory of the dataset
        dataset (str): Dataset name, one of ['koniq', 'live']
        split (str): Data split, one of ['train', 'val', 'test']
        transform (callable, optional): Optional transform to be applied to images
    """
    def __init__(self, root, dataset, split='train', transform=None):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.transform = transform
        
        if dataset == 'koniq':
            self._init_koniq()
        elif dataset == 'kadid':
            self._init_kadid()
        elif dataset == 'agiqa':
            self._init_agiqa()
        else:
            raise ValueError(f"Dataset {dataset} not supported.")
            
    def _init_koniq(self):
        """Initialize KonIQ-10k dataset."""
        # Path to KonIQ-10k CSV file with annotations
        csv_path = os.path.join(self.root, 'koniq10k', 'koniq10k_distributions_sets.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"KonIQ-10k CSV file not found at {csv_path}")
            
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Select data based on split
        if self.split == 'train':
            df = df[df['set'] == 'training']
        elif self.split == 'val':
            df = df[df['set'] == 'validation']
        elif self.split == 'test':
            df = df[df['set'] == 'test']
            
        # Get image paths and quality scores
        self.image_paths = [os.path.join(self.root, 'koniq10k', '1024x768', row['image_name']) 
                           for _, row in df.iterrows()]
        self.quality_scores = df['MOS'].values / 100.0  # Normalize to [0, 1]

    def _init_kadid(self):
        """Initialize KADID-10k dataset."""
        # Path to KADID-10k CSV file with annotations
        csv_path = os.path.join(self.root, 'kadid10k', 'split_kadid10k.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Kadid-10k CSV file not found at {csv_path}")
            
        # Read CSV
        df = pd.read_csv(csv_path)

        # Return a fraction, just for testing
        if self.split == 'train':
            df = df[df['split'] == 'train']
        elif self.split == 'val':
            df = df[df['split'] == 'val']
        elif self.split == 'test':
            df = df[df['split'] == 'test']
        
        # Get image paths and quality scores
        self.image_paths = [os.path.join(self.root, 'kadid10k', 'images', row['dist_img']) 
                           for _, row in df.iterrows()]
        self.quality_scores = df['dmos'].values / 5.0  # Normalize to [0, 1]
        

    def _init_agiqa(self):
        """Initialize AGIQA3K dataset."""
        # Path to AGIQA3K CSV file with annotations
        csv_path = os.path.join(self.root, 'agiqa3k', 'split_agiqa3k.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"AGIQA3k CSV file not found at {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)

        if self.split == 'train':
            df = df[df['split'] == 'train']
        elif self.split == 'val':
            df = df[df['split'] == 'val']
        elif self.split == 'test':
            df = df[df['split'] == 'test']
        
        # Get image paths and quality scores
        self.image_paths = [os.path.join(self.root, 'agiqa3k', 'images', row['name']) 
                           for _, row in df.iterrows()]
        self.quality_scores = df['mos_quality'].values / 5.0  # Normalize to [0, 1]

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)
            
        # Get quality score
        score = self.quality_scores[idx]
        
        return image, score 