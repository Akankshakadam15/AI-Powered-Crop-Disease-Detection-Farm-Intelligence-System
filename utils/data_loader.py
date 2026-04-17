import torch
import timm

MODEL_PATH = "plant_disease_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def verify_and_load():
    print(f"Device: {DEVICE}")
    print(f"Loading: {MODEL_PATH}\n")

    # Step 1: Inspect the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    print("--- Checkpoint Keys ---")
    print(list(checkpoint.keys()))

    num_classes = checkpoint.get('num_classes', None)
    class_names = checkpoint.get('class_names', None)

    print(f"\nNum Classes : {num_classes}")
    print(f"Class Names : {class_names[:5] if class_names else 'Not saved'} ...")

    # Step 2: Load into ResNet18
    print("\n--- Loading into ResNet18 ---")
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes or 38)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded successfully!")
    print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Test with a dummy input
    print("\n--- Running dummy inference ---")
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy)
    print(f"✅ Output shape: {output.shape}  — expected: torch.Size([1, {num_classes or 38}])")
    print("\n🎉 Everything looks good! You can now run: streamlit run app.py")

if __name__ == "__main__":
    verify_and_load()