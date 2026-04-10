# Alzheimer's Disease Diagnosis via Structural MRI
# Deep Learning Project for Early Detection using 3D CNNs

## Overview
This project develops a deep learning pipeline for automatic quantification of regional brain atrophy to predict Alzheimer's Disease (AD) risk from T1-weighted structural MRI scans.

## Dataset
Using OASIS-3 dataset (Open Access Series of Imaging Studies) with thousands of labeled MRI scans.

## Architecture
3D Convolutional Neural Network based on 3D-ResNet for multi-class classification:
- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Alzheimer's Disease (AD)

## Pipeline
1. Data Download and Exploration
2. Preprocessing (intensity normalization, skull-stripping, spatial normalization)
3. Model Training
4. Evaluation with Grad-CAM visualization
5. Inference Script

## Requirements
- Python 3.8+
- PyTorch
- nibabel
- numpy
- pandas
- matplotlib
- scikit-learn
- kagglehub

## Usage
1. Download data: `python src/download_data.py`
2. Preprocess: `python src/preprocess.py`
3. Train: `python src/train.py`
4. Evaluate: `python src/evaluate.py`
5. Infer: `python src/infer.py --input path/to/mri.nii.gz`

## Deliverables
- PDF Report with metrics and visualizations
- Training code optimized for GPU clusters
- Pre-trained model
- Inference script