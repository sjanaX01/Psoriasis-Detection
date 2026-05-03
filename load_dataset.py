"""
Dataset Loader for Psoriasis Detection
Loads train, test, and validation datasets from metadata.csv
"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
# from timm.data import create_transform
import cv2

class PsoriasisDataset(Dataset):
    """Custom Dataset for loading psoriasis images"""

    def __init__(self, dataframe, img_dir, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.iloc[idx]['image_path']
        label        = self.df.iloc[idx]['label_id']
        img_path     = self.img_dir / img_rel_path

        # ── Load as numpy (Albumentations চায়) ──
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Error loading image {img_path}, using blank image")
            image = np.ones((224, 224, 3), dtype=np.uint8) * 255  # white blank

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

        if self.transform:
            augmented = self.transform(image=image)
            image     = augmented['image']              # tensor [3, H, W]

        return image, label


def load_dataset(metadata_path='Dataset/metadata.csv', 
                 images_dir='Dataset/Images',
                 batch_size=32,
                 num_workers=4,
                 img_size=224):
    """
    Load train, validation, and test datasets
    
    Args:
        metadata_path: Path to metadata.csv file
        images_dir: Path to images directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        img_size: Image size for resizing
        
    Returns:
        train_loader, val_loader, test_loader, class_names, num_classes
    """
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Total images: {len(df)}")
    print(f"\nDataset split distribution:")
    print(df['split'].value_counts())
    print(f"\nNumber of classes: {df['label'].nunique()}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Split data
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Get class names and number of classes
    class_names = sorted(df['label'].unique())
    num_classes = len(class_names)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PsoriasisDataset(train_df, images_dir, transform=train_transform)
    val_dataset = PsoriasisDataset(val_df, images_dir, transform=val_test_transform)
    test_dataset = PsoriasisDataset(test_df, images_dir, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names, num_classes,val_dataset, train_dataset


def load_dataframes(metadata_path='Dataset/metadata.csv'):
    """
    Load metadata and return train, val, test dataframes separately
    
    Args:
        metadata_path: Path to metadata.csv file
        
    Returns:
        train_df, val_df, test_df, class_names, num_classes
    """
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Total images: {len(df)}")
    print(f"\nDataset split distribution:")
    print(df['split'].value_counts())
    
    # Split data
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Get class names
    class_names = sorted(df['label'].unique())
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    return train_df, val_df, test_df, class_names, num_classes


def build_dataset(dataframe, Transform, images_dir='Dataset/Images'):
    return PsoriasisDataset(dataframe, images_dir, transform=Transform)

def build_dataLoader(dataset: PsoriasisDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,       
                     num_workers: int = 4):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True 
    )

def build_transform(is_train, img_size, color_jitter, mag=0.7, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],crop_pct=None):

    if is_train:
        if img_size <= 32:
            return A.Compose([
                A.RandomCrop(img_size, img_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
            
        # Keep training augmentation conservative for medical images.
        # Preserve lesion color/texture while adding only mild geometric variation.
        train_ops = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=7,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=0,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.15
            ),
        ]

        if color_jitter is not None and color_jitter > 0:
            jitter = max(0.0, min(color_jitter, 0.15))
            train_ops.append(
                A.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter * 0.6,
                    hue=jitter * 0.1,
                    p=0.15
                )
            )

        train_ops.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        return A.Compose(train_ops)

    t = []

    if img_size >= 384:
        # বড় image — সরাসরি resize, no crop
        t.append(
            A.Resize(
                height=img_size,
                width=img_size,
                interpolation=cv2.INTER_CUBIC
            )
        )
        print(f"Warping {img_size} size input images...")

    else:
        # Standard — resize then center crop
        if crop_pct is None:
            crop_pct = 224 / 256        # timm default

        size = int(img_size / crop_pct) # e.g. 224/0.875 = 256

        t.append(
            A.Resize(
                height=size,
                width=size,
                interpolation=cv2.INTER_CUBIC
            )
        )
        t.append(
            A.CenterCrop(
                height=img_size,
                width=img_size
            )
        )

    t.append(A.Normalize(mean=mean, std=std))
    t.append(ToTensorV2())

    return A.Compose(t)
    

if __name__ == "__main__":
 
    train_df, val_df, test_df, class_names, num_classes = load_dataframes()
    
    train_transfromation = build_transform(is_train=True, img_size=224, color_jitter=0.4)
    train_dataset = build_dataset(train_df, Transform=train_transfromation)
    train_dataloader = build_dataLoader(train_dataset)
    
    img, lbl = next(iter(train_dataloader))
    print(f"Image size: {img.shape}")
    print(f"Label Size: {lbl.shape}")
