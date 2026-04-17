# Alzheimer's Disease Diagnosis - Final Report

## Executive Summary
This project implements a deep learning pipeline for early Alzheimer's Disease (AD) detection using structural MRI brain scans. Due to dataset availability, we adapted the approach to use 2D CNNs on MRI slices instead of the originally planned 3D CNNs on volumetric data.

## Dataset
- **Source**: OASIS dataset (via Kaggle)
- **Format**: 2D JPG slices from T1-weighted MRI scans
- **Classes**: 
  - Cognitively Normal (CN): 67,217 images
  - Mild Cognitive Impairment (MCI): 13,720 images  
  - Alzheimer's Disease (AD): 10,023 images
- **Total**: 86,437 images
- **Split**: 70% train, 20% validation, 10% test

## Model Architecture
- **Type**: 2D Convolutional Neural Network
- **Backbone**: ResNet-18 (adapted for single-channel input)
- **Input**: 224x224 grayscale images
- **Output**: 3-class classification (CN/MCI/AD)
- **Parameters**: ~11.2M trainable parameters

## Training Details
- **Framework**: PyTorch with PyTorch Lightning
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Epochs**: 5 (early stopping enabled)
- **Hardware**: CPU training (GPU recommended for production)

## Performance Metrics
[Note: Metrics will be populated after full training completion]

### Classification Report
```
              precision    recall  f1-score   support

          CN       0.XX      0.XX      0.XX      XXXX
         MCI       0.XX      0.XX      0.XX      XXXX
          AD       0.XX      0.XX      0.XX      XXXX

    accuracy                           0.XX      XXXX
   macro avg       0.XX      0.XX      0.XX      XXXX
weighted avg       0.XX      0.XX      0.XX      XXXX
```

### AUC-ROC Scores
- CN: X.XXX
- MCI: X.XXX  
- AD: X.XXX

## Grad-CAM Visualization
The model includes Grad-CAM implementation for interpretability, highlighting regions of the brain that contribute most to the classification decision.

## Recommendations for Production
1. **Dataset**: Obtain full 3D NIfTI volumes from OASIS-3 or ADNI for true volumetric analysis
2. **Hardware**: Train on GPU cluster (SUTD Mega Cluster or similar)
3. **Model**: Implement 3D CNN (VoxCNN or 3D-ResNet) for volumetric patterns
4. **Preprocessing**: Add skull-stripping, intensity normalization, and spatial registration
5. **Clinical Validation**: Validate with medical experts and larger datasets

## Usage Instructions
1. Download dataset: `python src/download_data.py`
2. Preprocess: `python src/preprocess.py`
3. Train: `python src/train.py`
4. Evaluate: `python src/evaluate.py`
5. Infer: `python src/infer.py --input path/to/mri_slice.jpg`

## Files Included
- `src/download_data.py`: Dataset download script
- `src/preprocess.py`: Data preprocessing and splitting
- `src/model.py`: 2D CNN model definition
- `src/train.py`: Training script with PyTorch Lightning
- `src/evaluate.py`: Model evaluation and metrics
- `src/infer.py`: Inference script with Grad-CAM
- `train_cluster.sh`: SLURM script for cluster training
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Future Work
- Implement 3D CNN pipeline with proper volumetric data
- Add data augmentation techniques
- Implement ensemble methods
- Clinical validation studies
- Deployment as web service for clinical use