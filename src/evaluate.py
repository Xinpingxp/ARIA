import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from torch.utils.data import DataLoader
from model import create_model
from preprocess import MRIImageDataset, get_data_loaders
from torchvision import transforms

def evaluate_model(model_path, data_splits_path, output_dir):
    """
    Evaluate the trained model and generate performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data splits
    with open(data_splits_path, 'r') as f:
        splits = json.load(f)

    # Load model
    model = create_model(num_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Build test loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = MRIImageDataset(
        splits['test']['images'],
        splits['test']['labels'],
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    class_names = ['CN', 'MCI', 'AD']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # AUC-ROC
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
        auc_scores[class_name] = auc
        print(f"AUC-ROC for {class_name}: {auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

    # Save metrics
    metrics = {
        'classification_report': report,
        'auc_scores': auc_scores,
        'confusion_matrix': cm.tolist()
    }

    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    data_splits_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'data_splits.json')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')

    # Find the best model
    models_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
    ckpt_files = list(models_dir.glob("*.ckpt"))
    if not ckpt_files:
        print("Model checkpoint not found. Run train.py first.")
    else:
        model_path = str(max(ckpt_files, key=lambda x: x.stat().st_mtime))
        evaluate_model(model_path, data_splits_path, output_dir)
