import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class MRIImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_data_splits(data_path, output_path, val_split=0.2, test_split=0.1):
    """
    Create train/val/test splits from the 2D image dataset.
    """
    os.makedirs(output_path, exist_ok=True)

    classes = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']
    class_mapping = {
        'Non Demented': 0,  # CN
        'Very mild Dementia': 1,  # MCI
        'Mild Dementia': 2,  # AD
        'Moderate Dementia': 2   # AD
    }

    all_images = []
    all_labels = []

    for cls in classes:
        class_path = os.path.join(data_path, 'Data', cls)
        if os.path.exists(class_path):
            image_files = list(Path(class_path).glob("*.jpg"))
            labels = [class_mapping[cls]] * len(image_files)
            all_images.extend(image_files)
            all_labels.extend(labels)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # Stratified train+val / test split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
    trainval_idx, test_idx = next(sss_test.split(all_images, all_labels))

    # Stratified train / val split
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split / (1 - test_split), random_state=42)
    train_idx, val_idx = next(sss_val.split(all_images[trainval_idx], all_labels[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    train_images = all_images[train_idx].tolist()
    train_labels = all_labels[train_idx].tolist()
    val_images = all_images[val_idx].tolist()
    val_labels = all_labels[val_idx].tolist()
    test_images = all_images[test_idx].tolist()
    test_labels = all_labels[test_idx].tolist()

    # Save splits
    splits = {
        'train': {'images': [str(p) for p in train_images], 'labels': train_labels},
        'val': {'images': [str(p) for p in val_images], 'labels': val_labels},
        'test': {'images': [str(p) for p in test_images], 'labels': test_labels}
    }

    import json
    with open(os.path.join(output_path, 'data_splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Data splits created:")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

    return splits

def get_data_loaders(data_splits, batch_size=32, image_size=224):
    """
    Create PyTorch data loaders for train/val/test.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MRIImageDataset(
        data_splits['train']['images'],
        data_splits['train']['labels'],
        transform=train_transform
    )

    val_dataset = MRIImageDataset(
        data_splits['val']['images'],
        data_splits['val']['labels'],
        transform=val_test_transform
    )

    test_dataset = MRIImageDataset(
        data_splits['test']['images'],
        data_splits['test']['labels'],
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'oasis')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    if os.path.exists(data_path):
        splits = create_data_splits(data_path, output_path)
        train_loader, val_loader, test_loader = get_data_loaders(splits)

        # Test loaders
        print("Testing data loaders...")
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
            break
    else:
        print("Raw dataset not found. Run download_data.py first.")