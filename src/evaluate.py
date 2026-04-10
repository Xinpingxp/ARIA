import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pandas as pd
from torch.utils.data import DataLoader
from model import create_model
from train import MRIDataset

def evaluate_model(model_path, test_data_path, output_dir):
    """
    Evaluate the trained model and generate performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = create_model(num_classes=3)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load test data
    metadata_file = os.path.join(test_data_path, "processed_metadata.csv")
    df = pd.read_csv(metadata_file)

    # For demo, use same dummy labels as training
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=len(df))

    # Use last 10% as test
    test_size = int(0.1 * len(df))
    test_data = df[-test_size:]
    test_labels = labels[-test_size:]

    test_dataset = MRIDataset(test_data['processed_path'].tolist(), test_labels)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
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

    import json
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.ckpt')  # Adjust path
    test_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')

    # Find the best model if not specified
    models_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
    ckpt_files = list(models_dir.glob("*.ckpt"))
    if ckpt_files:
        model_path = str(max(ckpt_files, key=lambda x: x.stat().st_mtime))

    if os.path.exists(model_path):
        evaluate_model(model_path, test_data_path, output_dir)
    else:
        print("Model checkpoint not found. Run train.py first.")