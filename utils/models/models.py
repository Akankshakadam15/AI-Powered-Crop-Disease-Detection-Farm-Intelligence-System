i@st.cache_resource
def load_model():
    # Step 1: Load the checkpoint dict
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Step 2: Get num_classes from checkpoint (or fallback to 38)
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)

    # Step 3: Use ResNet18 — same as training!
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)

    # Step 4: Load only the model weights (not the full dict)
    model.load_state_dict(checkpoint['model_state'])

    model.to(DEVICE)
    model.eval()
    return model