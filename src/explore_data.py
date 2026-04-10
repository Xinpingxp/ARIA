import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def explore_dataset(data_path):
    """
    Explore the downloaded OASIS dataset structure and metadata.
    Note: This dataset contains 2D JPG slices, not 3D NIfTI volumes.
    """
    print(f"Exploring dataset at: {data_path}")

    # List contents
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")

def load_sample_images(data_path):
    """
    Load and display sample images from each class.
    """
    classes = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']
    class_mapping = {
        'Non Demented': 'CN',
        'Very mild Dementia': 'MCI',
        'Mild Dementia': 'AD',
        'Moderate Dementia': 'AD'
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for i, cls in enumerate(classes):
        class_path = os.path.join(data_path, 'Data', cls)
        if os.path.exists(class_path):
            jpg_files = list(Path(class_path).glob("*.jpg"))
            if jpg_files:
                sample_file = jpg_files[0]
                img = Image.open(sample_file)
                img_array = np.array(img)

                axes[i].imshow(img_array, cmap='gray')
                axes[i].set_title(f"{cls} ({class_mapping[cls]}) - {img_array.shape}")
                axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'reports', 'sample_images.png'))
    plt.show()

    # Count images per class
    print("\nImage counts per class:")
    total_images = 0
    for cls in classes:
        class_path = os.path.join(data_path, 'Data', cls)
        if os.path.exists(class_path):
            count = len(list(Path(class_path).glob("*.jpg")))
            print(f"{cls}: {count} images")
            total_images += count

    print(f"Total images: {total_images}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'oasis')
    if os.path.exists(data_path):
        explore_dataset(data_path)
        load_sample_images(data_path)
    else:
        print("Dataset not found. Run download_data.py first.")