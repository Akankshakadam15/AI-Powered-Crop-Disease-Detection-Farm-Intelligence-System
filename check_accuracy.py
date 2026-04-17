# -*- coding: utf-8 -*-
"""
Model Accuracy Check Script
Usage: python check_accuracy.py
"""

import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# ============ CLASS NAMES ============
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

# ============ PATHS ============
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "plant_disease_model.pth")
DATA_DIR    = os.path.join(BASE_DIR, "data", "test")

# ============ DEVICE ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")

# ============ LOAD MODEL ============
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    model = timm.create_model('resnet18', pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded!")
    return model

# ============ CHECK ACCURACY ============
def check_accuracy(model):
    if not os.path.exists(DATA_DIR):
        print("❌ data/test folder not found!")
        print("   Use predict.py to test a single image.")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset    = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"\n📊 Test Images: {len(dataset)}")
    print(f"   Classes    : {len(dataset.classes)}")
    print(f"\n⏳ Calculating accuracy...")

    correct, total = 0, 0
    class_correct  = [0] * len(CLASS_NAMES)
    class_total    = [0] * len(CLASS_NAMES)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for label, pred in zip(labels, preds):
                class_total[label.item()]   += 1
                class_correct[label.item()] += (pred == label).item()

            if (i + 1) % 10 == 0:
                print(f"   Batch {i+1}/{len(dataloader)} done...")

    overall_acc = correct / total * 100
    print(f"\n{'='*50}")
    print(f"🏆 Overall Accuracy : {overall_acc:.2f}%")
    print(f"   Correct          : {correct}/{total}")
    print(f"{'='*50}")

    # Per-class accuracy
    print(f"\n📋 Per-Class Accuracy:")
    print(f"{'Class':<50} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 78)
    for i, name in enumerate(dataset.classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"{name:<50} {acc:>9.2f}%  {class_correct[i]:>5}/{class_total[i]:<5}")

# ============ MAIN ============
if __name__ == "__main__":
    model = load_model()
    check_accuracy(model)