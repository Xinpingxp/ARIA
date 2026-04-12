import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
from PIL import Image
from model import create_model
from torchvision import transforms

class GradCAM2D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Hook to capture gradients
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_maps = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)  # Normalize

        return cam.squeeze().detach().cpu().numpy()

def infer_ad_risk(image_path, model_path, output_dir):
    """
    Perform inference on a new MRI slice and generate risk assessment.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert('L')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load model
    model = create_model(num_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        prediction = torch.argmax(output, dim=1).item()

    class_names = ['Cognitively Normal (CN)', 'Mild Cognitive Impairment (MCI)', 'Alzheimer\'s Disease (AD)']
    predicted_class = class_names[prediction]

    print("Inference Results:")
    print(f"Predicted Class: {predicted_class}")
    print("Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob:.3f}")

    # Generate Grad-CAM
    grad_cam = GradCAM2D(model, model.backbone.layer4)
    cam = grad_cam.generate_cam(input_tensor, prediction)

    # Visualize CAM
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original MRI Slice')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Attention Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'))
    plt.show()

    # Save results
    results = {
        'predicted_class': predicted_class,
        'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)},
        'risk_score': float(probabilities[2])  # AD probability as risk score
    }

    import json
    with open(os.path.join(output_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AD Risk Inference from MRI Slice')
    parser.add_argument('--input', type=str, required=True, help='Path to input MRI slice (.jpg or .png)')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='inference_output', help='Output directory')

    args = parser.parse_args()

    # Find model if not specified
    if not args.model:
        models_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'models'))
        ckpt_files = list(models_dir.glob("*.ckpt"))
        if ckpt_files:
            args.model = str(max(ckpt_files, key=lambda x: x.stat().st_mtime))
        else:
            raise FileNotFoundError("No model checkpoint found. Train the model first.")

    infer_ad_risk(args.input, args.model, args.output)