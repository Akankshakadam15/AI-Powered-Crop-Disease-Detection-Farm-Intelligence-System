# -*- coding: utf-8 -*-
"""
Plant Disease Detection - Prediction Script
Colab trained model साठी fix केलेला version
Usage: python predict.py --image path/to/your/image.jpg
"""

import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import argparse

# ============ PlantVillage Dataset चे Class Names ============
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ============ DEVICE ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ TRANSFORM ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============ LOAD MODEL ============
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model सापडला नाही: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Colab model - फक्त model_state आहे
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    num_classes = len(CLASS_NAMES)
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print(f"✅ Model loaded | Classes: {num_classes} | Device: {DEVICE}")
    return model

# ============ PREDICT ============
def predict(image_path, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image सापडली नाही: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()
        conf    = probs[pred].item() * 100

    print(f"\n🌿 Image   : {image_path}")
    print(f"   Result  : {CLASS_NAMES[pred]}")
    print(f"   Confidence: {conf:.2f}%")

    # Top 3
    top3 = probs.topk(3)
    print("\n   Top 3 Predictions:")
    for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
        print(f"   {i+1}. {CLASS_NAMES[idx.item()]:50s} {prob.item()*100:.2f}%")

    return CLASS_NAMES[pred], conf

# ============ MAIN ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    parser.add_argument("--image", required=True, help="Plant image चा path द्या")
    parser.add_argument("--model", default="plant_disease_model.pth",
                        help="Model file (default: plant_disease_model.pth)")
    args = parser.parse_args()

    model = load_model(args.model)
    predict(args.image, model)