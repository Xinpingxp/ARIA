import kagglehub
import os
import shutil

def download_oasis_dataset():
    """
    Download the OASIS dataset from Kaggle.
    This dataset contains T1-weighted MRI scans for Alzheimer's research.
    """
    print("Downloading OASIS dataset...")
    path = kagglehub.dataset_download("ninadaithal/imagesoasis")
    print("Path to dataset files:", path)

    # Move to data directory
    dest_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'oasis')
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    shutil.move(path, dest_path)

    print(f"Dataset moved to: {dest_path}")
    return dest_path

if __name__ == "__main__":
    download_oasis_dataset()