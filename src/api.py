from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import base64
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import create_model
from infer import GradCAM2D

app = Flask(__name__)
CORS(app)

MODELS_DIR = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
CLASS_NAMES = ['Cognitively Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]

# --- Model loading ---

def find_latest_checkpoint():
    if not MODELS_DIR.exists():
        return None
    ckpt_files = list(MODELS_DIR.glob("*.ckpt"))
    if not ckpt_files:
        return None
    return str(max(ckpt_files, key=lambda x: x.stat().st_mtime))

def load_model(model_path):
    m = create_model(num_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    # Lightning saves weights with a 'model.' prefix — strip it
    state_dict = {
        k.replace('model.', '', 1): v
        for k, v in checkpoint['state_dict'].items()
    }
    m.load_state_dict(state_dict)
    m.eval()
    return m

_model = None
_model_path = find_latest_checkpoint()
if _model_path:
    try:
        _model = load_model(_model_path)
        print(f"Model loaded: {_model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
else:
    print("No model checkpoint found — running in demo mode.")

# --- Helpers ---

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor, image

def heatmap_to_base64(original_image, cam):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_image, cmap='gray')
    ax.imshow(cam, cmap='jet', alpha=0.5,
              extent=[0, original_image.width, original_image.height, 0])
    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- Routes ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_bytes = file.read()

    # Demo mode — no model checkpoint available
    if _model is None:
        import random
        random.seed(42)
        cn  = round(random.uniform(2, 8), 1)
        mci = round(random.uniform(18, 28), 1)
        ad  = round(100 - cn - mci, 1)
        return jsonify({
            'demo_mode': True,
            'predicted_class': "Alzheimer's Disease",
            'probabilities': {'cn': cn, 'mci': mci, 'ad': ad},
            'heatmap': None,
            'message': 'Demo mode — no trained model found. Add a .ckpt file to models/ for real predictions.'
        })

    try:
        input_tensor, original_image = preprocess_image(image_bytes)

        with torch.no_grad():
            output = _model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            prediction = int(torch.argmax(output, dim=1).item())

        # Grad-CAM
        grad_cam = GradCAM2D(_model, _model.backbone.layer4)
        cam = grad_cam.generate_cam(input_tensor, prediction)
        heatmap_b64 = heatmap_to_base64(original_image, cam)

        return jsonify({
            'demo_mode': False,
            'predicted_class': CLASS_NAMES[prediction],
            'probabilities': {
                'cn':  round(float(probs[0]) * 100, 1),
                'mci': round(float(probs[1]) * 100, 1),
                'ad':  round(float(probs[2]) * 100, 1),
            },
            'heatmap': heatmap_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
