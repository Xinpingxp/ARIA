# ARIA: Automated Radiology Intelligence Assistant
# Alzheimer's Disease Detection from Brain MRI Slices

## Overview
ARIA is a deep learning pipeline for classifying Alzheimer's Disease risk from 2D T1-weighted brain MRI slices. The model classifies each slice into one of three categories: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), or Alzheimer's Disease (AD).

## Dataset
- **Source**: OASIS dataset via Kaggle ([ninadaithal/imagesoasis](https://www.kaggle.com/datasets/ninadaithal/imagesoasis))
- **Format**: 2D JPG slices from T1-weighted MRI scans
- **Classes**: Non Demented (CN), Very Mild Dementia (MCI), Mild/Moderate Dementia (AD)
- **Total**: ~86,000 images
- **Split**: 70% train, 20% validation, 10% test (stratified)

## Model Architecture
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Input**: 224x224 single-channel (grayscale) images
- **Head**: Dropout(0.5) + Linear(512, 3)
- **Output**: 3-class classification (CN / MCI / AD)

## Results
| Class | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| CN    | 0.988     | 0.906  | 0.946 | 0.982 |
| MCI   | 0.707     | 0.940  | 0.807 | 0.980 |
| AD    | 0.815     | 0.973  | 0.887 | 0.998 |

**Overall test accuracy: 91.59%**

## Requirements
- Python 3.9+
- See `requirements.txt` for full dependencies

```
pip install -r requirements.txt
```

## Usage

### 1. Download dataset
```
python src/download_data.py
```
Requires a Kaggle API token configured at `~/.kaggle/kaggle.json`. Alternatively, download manually from Kaggle and place in `data/oasis/`.

### 2. Preprocess
```
python src/preprocess.py
```

### 3. Train
```
python src/train.py
```
Automatically uses MPS (Apple Silicon), CUDA, or CPU depending on hardware available.

### 4. Evaluate
```
python src/evaluate.py
```
Outputs classification report, confusion matrix, AUC-ROC scores, and training curves to `reports/`.

### 5. Inference (single MRI slice)
```
python src/infer.py --input path/to/mri_slice.jpg
```
Outputs predicted class, probability scores, and a Grad-CAM attention heatmap.

## Model Checkpoint
The trained model checkpoint is tracked via Git LFS. After cloning, it will be available at:
```
models/ad-classifier-epoch=04-val_acc=0.92.ckpt
```

## Project Structure
```
ARIA/
├── src/
│   ├── download_data.py   # Kaggle dataset download
│   ├── preprocess.py      # Data splitting and augmentation
│   ├── model.py           # ResNet-18 model definition
│   ├── train.py           # PyTorch Lightning training loop
│   ├── evaluate.py        # Evaluation metrics and plots
│   └── infer.py           # Inference with Grad-CAM
├── models/                # Saved checkpoints
├── reports/               # Evaluation outputs
├── dashboard/             # React/Vite frontend
├── requirements.txt
└── README.md
```

## Frontend Dashboard

A React-based diagnostic interface for uploading MRI scans and viewing AI predictions.

### Tech Stack
- **Framework**: React 19
- **Build Tool**: Vite 8
- **Backend**: Flask API (proxied via Vite dev server)

### Features
- Drag-and-drop or click-to-upload MRI scans (.jpg, .png)
- Real-time AI inference with loading states
- Grad-CAM attention heatmap visualization
- Classification probabilities for all three classes (CN / MCI / AD)
- Risk assessment display (Low / Moderate / High)
- Demo mode when no trained model is available

### Running the Frontend

**1. Start the backend API**
```bash
python src/api.py
```
The Flask server runs on `http://localhost:5001`.

**2. Start the frontend dev server**
```bash
cd dashboard
npm install
npm run dev
```
The Vite dev server runs on `http://localhost:5173` and proxies `/api` requests to the Flask backend.

**3. Build for production**
```bash
cd dashboard
npm run build
```
Output is generated in `dashboard/dist/`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns model load status |
| `/predict` | POST | Upload MRI image, returns predictions and Grad-CAM heatmap |

### Screenshots

| Non-Demented | Very Mild Dementia |
|:---:|:---:|
| ![Non-Demented](FINAL%20FRONTEND%20IMAGE/non-demented.png) | ![Very Mild](FINAL%20FRONTEND%20IMAGE/very-mild%20dementia.png) |

| Mild Dementia | Moderate Dementia |
|:---:|:---:|
| ![Mild](FINAL%20FRONTEND%20IMAGE/mild-dementia.png) | ![Moderate](FINAL%20FRONTEND%20IMAGE/moderate%20dementia.png) |
