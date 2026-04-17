import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import os
import requests
from datetime import datetime
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

# ============ CONFIG ============
MODEL_PATH  = "plant_disease_model.pth"
NUM_CLASSES = 38
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenWeatherMap API Key - apla key yethhe ghala
WEATHER_API_KEY = "YOUR_API_KEY_HERE"  # https://openweathermap.org/api madhe free account kara

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DISEASE_INFO = {
    "Apple_scab"            : ("Fungal disease causing dark scabs on leaves and fruit.",
                               "🌿 Use fungicide sprays. Remove infected leaves. Avoid overhead watering."),
    "Black_rot"             : ("Fungal disease causing black rotting of fruit and leaves.",
                               "✂️ Prune infected branches. Apply copper-based fungicide. Destroy fallen fruit."),
    "Cedar_apple_rust"      : ("Fungal disease causing orange spots on leaves.",
                               "🍎 Remove nearby cedar trees. Use myclobutanil fungicide in spring."),
    "Powdery_mildew"        : ("Fungal disease causing white powdery coating on leaves.",
                               "💨 Improve air circulation. Apply sulfur-based fungicide. Avoid wet leaves."),
    "Cercospora_leaf_spot"  : ("Fungal disease causing gray spots on leaves.",
                               "🌾 Rotate crops. Apply chlorothalonil fungicide. Remove crop debris."),
    "Common_rust"           : ("Fungal disease causing rust-colored pustules on leaves.",
                               "🌽 Plant resistant varieties. Apply fungicide early. Monitor regularly."),
    "Northern_Leaf_Blight"  : ("Fungal disease causing large tan lesions on leaves.",
                               "🌿 Use resistant hybrids. Apply triazole fungicide at early stages."),
    "Bacterial_spot"        : ("Bacterial disease causing small dark spots on leaves.",
                               "🦠 Use copper bactericide. Avoid overhead irrigation. Remove infected plant parts."),
    "Early_blight"          : ("Fungal disease causing dark spots with rings on leaves.",
                               "🍅 Apply mancozeb fungicide. Mulch soil. Water at base of plant."),
    "Late_blight"           : ("Serious disease causing water-soaked lesions on leaves and stems.",
                               "⚠️ Apply metalaxyl fungicide immediately. Remove infected plants. Improve drainage."),
    "Leaf_Mold"             : ("Fungal disease causing yellow spots on upper leaf surface.",
                               "🍃 Reduce humidity. Improve ventilation. Apply chlorothalonil fungicide."),
    "Septoria_leaf_spot"    : ("Fungal disease causing small circular spots with dark borders.",
                               "🌿 Remove lower infected leaves. Apply fungicide. Avoid wetting foliage."),
    "Spider_mites"          : ("Pest causing stippling and webbing on leaves.",
                               "🕷️ Apply miticide or neem oil. Increase humidity. Introduce predatory mites."),
    "Target_Spot"           : ("Fungal disease causing target-like circular spots on leaves.",
                               "🎯 Apply azoxystrobin fungicide. Rotate crops. Remove infected debris."),
    "Yellow_Leaf_Curl_Virus": ("Viral disease causing yellowing and curling of leaves.",
                               "🦟 Control whitefly vectors. Use reflective mulches. Remove infected plants."),
    "mosaic_virus"          : ("Viral disease causing mosaic pattern on leaves.",
                               "🧬 No cure. Remove infected plants. Control aphid vectors. Use virus-free seeds."),
    "Haunglongbing"         : ("Serious citrus disease spread by psyllid insects.",
                               "🍊 Remove infected trees. Control psyllid population. Use certified disease-free plants."),
    "Esca"                  : ("Fungal complex causing internal wood decay in grapevines.",
                               "🍇 Prune infected wood. Apply wound protectants. Avoid pruning in wet weather."),
    "Leaf_scorch"           : ("Fungal disease causing browning and scorching of leaf margins.",
                               "🍓 Apply appropriate fungicide. Ensure proper irrigation. Remove infected leaves."),
    "healthy"               : ("Plant appears healthy with no visible signs of disease.",
                               "✅ Continue regular monitoring. Maintain proper irrigation and fertilization."),
}

# ============ SESSION STATE INIT ============
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'disease_count' not in st.session_state:
    st.session_state.disease_count = 0
if 'healthy_count' not in st.session_state:
    st.session_state.healthy_count = 0

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ['model_state_dict', 'state_dict', 'model_state', 'model', 'net']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        if state_dict is None:
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                num_classes = NUM_CLASSES
            else:
                raise KeyError(f"Weights not found. Keys: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint
        num_classes = NUM_CLASSES

    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ============ GRAD-CAM ============
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def generate_gradcam_image(model, image, class_idx):
    try:
        target_layer = model.layer4[-1]
        grad_cam = GradCAM(model, target_layer)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)
        cam = grad_cam.generate(input_tensor, class_idx)
        img_resized = image.resize((224, 224))
        img_array   = np.array(img_resized)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap     = cm.jet(cam_resized)[:, :, :3]
        heatmap     = (heatmap * 255).astype(np.uint8)
        overlay     = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception as e:
        return None

# ============ PREDICT ============
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)
        top3   = torch.topk(probs, 3)
    results = []
    for i in range(3):
        idx  = top3.indices[0][i].item()
        prob = top3.values[0][i].item() * 100
        results.append((CLASS_NAMES[idx], prob, idx))
    return results

def get_disease_info(class_name):
    for key, (info, treatment) in DISEASE_INFO.items():
        if key.lower() in class_name.lower():
            return info, treatment
    return "No additional information available.", "Consult an agricultural expert."

def get_severity(confidence):
    if confidence >= 90:
        return "🔴 High", "red"
    elif confidence >= 70:
        return "🟠 Medium", "orange"
    else:
        return "🟡 Low", "yellow"

# ============ WEATHER ============
def get_weather(city="Aurangabad"):
    if WEATHER_API_KEY == "YOUR_API_KEY_HERE":
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        r   = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def get_disease_risk(weather_data):
    if not weather_data:
        return None
    temp     = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    risks    = []
    if humidity > 80 and temp > 20:
        risks.append(("🍄 Fungal Disease Risk", "HIGH", "red",
                      "High humidity + warm temp = ideal conditions for fungal diseases like blight and mildew."))
    elif humidity > 65:
        risks.append(("🍄 Fungal Disease Risk", "MEDIUM", "orange",
                      "Moderate humidity - monitor crops closely for early signs of fungal infection."))
    else:
        risks.append(("🍄 Fungal Disease Risk", "LOW", "green",
                      "Current conditions are not favorable for most fungal diseases."))

    if temp > 28:
        risks.append(("🦟 Pest Activity Risk", "HIGH", "red",
                      "High temperature increases insect activity. Watch for spider mites and aphids."))
    elif temp > 22:
        risks.append(("🦟 Pest Activity Risk", "MEDIUM", "orange",
                      "Moderate pest risk. Regular inspection recommended."))
    else:
        risks.append(("🦟 Pest Activity Risk", "LOW", "green",
                      "Low temperature reduces pest activity risk."))

    if humidity > 75 and temp > 15:
        risks.append(("🦠 Bacterial Disease Risk", "MEDIUM", "orange",
                      "Warm and humid conditions can favor bacterial diseases. Avoid overhead irrigation."))
    else:
        risks.append(("🦠 Bacterial Disease Risk", "LOW", "green",
                      "Current conditions show low bacterial disease risk."))
    return risks

# ============ UI SETUP ============
st.set_page_config(
    page_title="SmartAgriGuard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1B5E20, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .healthy {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-left: 6px solid #4CAF50;
    }
    .diseased {
        background: linear-gradient(135deg, #FFEBEE, #FFF3E0);
        border-left: 6px solid #F44336;
    }
    .info-box {
        background: linear-gradient(135deg, #E3F2FD, #EDE7F6);
        border-left: 6px solid #2196F3;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .treatment-box {
        background: linear-gradient(135deg, #F3E5F5, #E8EAF6);
        border-left: 6px solid #9C27B0;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .weather-card {
        background: linear-gradient(135deg, #E1F5FE, #E0F7FA);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .stat-card {
        background: linear-gradient(135deg, #F5F5F5, #FAFAFA);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .risk-high   { background-color: #FFEBEE; border-left: 5px solid #F44336;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .risk-medium { background-color: #FFF8E1; border-left: 5px solid #FF9800;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .risk-low    { background-color: #E8F5E9; border-left: 5px solid #4CAF50;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown('<p class="main-title">🌾 SmartAgriGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Crop Disease Detection & Farm Intelligence System</p>',
            unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🌿 SmartAgriGuard")
    st.markdown("---")

    # Location Input for Weather
    st.subheader("📍 Your Location")
    city = st.text_input("City Name", value="Aurangabad", help="Enter your city for weather & risk analysis")

    st.markdown("---")
    st.subheader("📊 Session Stats")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_scans}</div><small>Scans</small></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#F44336">{st.session_state.disease_count}</div><small>Diseased</small></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.healthy_count}</div><small>Healthy</small></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("ℹ️ Supported Plants")
    plants = ["🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn",
              "🍇 Grape", "🍊 Orange", "🍑 Peach", "🫑 Pepper",
              "🥔 Potato", "🫙 Raspberry", "🫘 Soybean", "🎃 Squash",
              "🍓 Strawberry", "🍅 Tomato"]
    for p in plants:
        st.write(p)

    st.markdown("---")
    st.caption("Model: ResNet18 | Accuracy: ~95% | Classes: 38")

# ============ MAIN TABS ============
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Disease Detection", "🌦️ Weather & Risk", "📊 Dashboard", "📋 Scan History"])

# ===== TAB 1 - DETECTION =====
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        show_gradcam = st.checkbox("🔥 Show Grad-CAM Heatmap", value=True,
                                   help="Highlights which part of the leaf the AI is focusing on")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📸 Uploaded Leaf Image", use_column_width=True)
            st.success(f"✅ Image loaded: {uploaded_file.name}")

    with col2:
        st.subheader("🔍 Analysis Results")

        if uploaded_file is None:
            st.info("👈 Upload a leaf image to get started!")
            st.markdown("""
            **How to use:**
            1. 📸 Click 'Browse files' and upload a leaf image
            2. 🔬 AI will detect disease automatically
            3. 💊 Get treatment recommendations
            4. 🔥 View Grad-CAM heatmap (where AI is looking)
            """)
        else:
            if not os.path.exists(MODEL_PATH):
                st.error("❌ Model file not found! Place `plant_disease_model.pth` in project folder.")
            else:
                with st.spinner("🔬 Analyzing leaf with AI..."):
                    model   = load_model()
                    results = predict(image, model)

                top_class, top_prob, top_idx = results[0]
                parts     = top_class.split("___")
                plant     = parts[0].replace("_", " ")
                condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                is_healthy = "healthy" in condition.lower()

                # Update session stats
                st.session_state.total_scans += 1
                if is_healthy:
                    st.session_state.healthy_count += 1
                else:
                    st.session_state.disease_count += 1

                # Main result
                if is_healthy:
                    st.markdown(f"""
                    <div class="result-box healthy">
                        <h3>✅ Plant is Healthy!</h3>
                        <p><b>🌱 Plant:</b> {plant}</p>
                        <p><b>📊 Confidence:</b> {top_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    severity_label, severity_color = get_severity(top_prob)
                    st.markdown(f"""
                    <div class="result-box diseased">
                        <h3>⚠️ Disease Detected!</h3>
                        <p><b>🌱 Plant:</b> {plant}</p>
                        <p><b>🦠 Disease:</b> {condition}</p>
                        <p><b>📊 Confidence:</b> {top_prob:.1f}%</p>
                        <p><b>⚡ Severity:</b> {severity_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Disease info & treatment
                info, treatment = get_disease_info(top_class)
                st.markdown(f'<div class="info-box"><b>📋 About:</b><br>{info}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="treatment-box"><b>💊 Treatment:</b><br>{treatment}</div>',
                            unsafe_allow_html=True)

                # Top 3 predictions
                st.write("**📊 Top 3 Predictions:**")
                for i, (cls, prob, _) in enumerate(results):
                    p    = cls.split("___")[0].replace("_", " ")
                    c    = cls.split("___")[1].replace("_", " ") if "___" in cls else ""
                    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.progress(int(prob) / 100)
                    st.write(f"{icon} **{p}** — {c} ({prob:.1f}%)")

                # Grad-CAM
                if show_gradcam:
                    st.write("**🔥 Grad-CAM Heatmap (AI Focus Area):**")
                    with st.spinner("Generating heatmap..."):
                        model_grad = load_model()
                        cam_image  = generate_gradcam_image(model_grad, image, top_idx)
                    if cam_image:
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.image(image.resize((224, 224)), caption="Original", use_column_width=True)
                        with gcol2:
                            st.image(cam_image, caption="AI Focus (Red = High Attention)", use_column_width=True)
                        st.caption("🔴 Red areas = Where AI detected disease patterns")
                    else:
                        st.info("Grad-CAM not available for this image.")

                # Save to history
                st.session_state.history.append({
                    "time"      : datetime.now().strftime("%H:%M:%S"),
                    "date"      : datetime.now().strftime("%d/%m/%Y"),
                    "plant"     : plant,
                    "condition" : condition,
                    "confidence": f"{top_prob:.1f}%",
                    "status"    : "Healthy" if is_healthy else "Diseased"
                })

# ===== TAB 2 - WEATHER =====
with tab2:
    st.subheader("🌦️ Weather & Disease Risk Analysis")

    weather_data = get_weather(city)

    if weather_data is None and WEATHER_API_KEY == "YOUR_API_KEY_HERE":
        st.warning("⚠️ Weather API key not set! OpenWeatherMap madhe free account kara aani sidebar madhe city enter kara.")
        st.info("👉 https://openweathermap.org/api - Free API key milel. Nanthar WEATHER_API_KEY variable madhe ghala.")

        st.markdown("### 📖 Demo Mode - Sample Risk Analysis")
        st.markdown("""
        <div class="risk-high">
            <b>🍄 Fungal Disease Risk: HIGH</b><br>
            <small>High humidity (85%) + warm temperature (28°C) = ideal conditions for blight and mildew.</small>
        </div>
        <div class="risk-medium">
            <b>🦟 Pest Activity Risk: MEDIUM</b><br>
            <small>Moderate temperature increases insect activity. Watch for spider mites.</small>
        </div>
        <div class="risk-low">
            <b>🦠 Bacterial Disease Risk: LOW</b><br>
            <small>Current conditions show low bacterial disease risk.</small>
        </div>
        """, unsafe_allow_html=True)
    elif weather_data:
        # Weather Cards
        temp     = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        desc     = weather_data['weather'][0]['description'].title()
        wind     = weather_data['wind']['speed']
        feels    = weather_data['main']['feels_like']

        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            st.markdown(f'<div class="weather-card">🌡️<br><b>{temp:.1f}°C</b><br><small>Temperature</small></div>',
                        unsafe_allow_html=True)
        with w2:
            st.markdown(f'<div class="weather-card">💧<br><b>{humidity}%</b><br><small>Humidity</small></div>',
                        unsafe_allow_html=True)
        with w3:
            st.markdown(f'<div class="weather-card">🌤️<br><b>{desc}</b><br><small>Condition</small></div>',
                        unsafe_allow_html=True)
        with w4:
            st.markdown(f'<div class="weather-card">💨<br><b>{wind} m/s</b><br><small>Wind Speed</small></div>',
                        unsafe_allow_html=True)
        with w5:
            st.markdown(f'<div class="weather-card">🤔<br><b>{feels:.1f}°C</b><br><small>Feels Like</small></div>',
                        unsafe_allow_html=True)

        st.markdown("### 🎯 Disease Risk Assessment")
        risks = get_disease_risk(weather_data)
        if risks:
            for name, level, color, explanation in risks:
                css_class = f"risk-{level.lower()}"
                st.markdown(f"""
                <div class="{css_class}">
                    <b>{name}: {level}</b><br>
                    <small>{explanation}</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 💡 Today's Farming Recommendations")
        if humidity > 80:
            st.warning("⚠️ High humidity detected! Avoid overhead irrigation. Ensure good air circulation in crops.")
        if temp > 30:
            st.warning("🌡️ High temperature! Water crops in early morning or evening. Watch for pest activity.")
        if humidity < 40:
            st.info("💧 Low humidity. Increase irrigation frequency. Monitor for spider mite activity.")
        if humidity <= 80 and temp <= 30:
            st.success("✅ Weather conditions are favorable for most crops today!")
    else:
        st.error("❌ Could not fetch weather data. Check city name and API key.")

# ===== TAB 3 - DASHBOARD =====
with tab3:
    st.subheader("📊 Farm Analytics Dashboard")

    if st.session_state.total_scans == 0:
        st.info("📸 No scans yet! Go to Disease Detection tab and scan some leaves.")
    else:
        # Summary stats
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.total_scans}</div>
                <div>Total Scans</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:#F44336">{st.session_state.disease_count}</div>
                <div>Diseases Found</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state.healthy_count}</div>
                <div>Healthy Plants</div>
            </div>
            """, unsafe_allow_html=True)
        with d4:
            rate = (st.session_state.healthy_count / st.session_state.total_scans * 100
                    if st.session_state.total_scans > 0 else 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{rate:.0f}%</div>
                <div>Health Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        if len(st.session_state.history) > 0:
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("**🥧 Disease vs Healthy Ratio**")
                fig, ax = plt.subplots(figsize=(5, 4))
                labels  = ['Diseased 🦠', 'Healthy ✅']
                sizes   = [st.session_state.disease_count, st.session_state.healthy_count]
                colors  = ['#EF5350', '#66BB6A']
                explode = (0.05, 0.05)
                if sum(sizes) > 0:
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                           autopct='%1.1f%%', shadow=True, startangle=90)
                    ax.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig)
                plt.close()

            with chart_col2:
                st.write("**🌿 Plants Scanned**")
                plant_counts = {}
                for h in st.session_state.history:
                    p = h['plant']
                    plant_counts[p] = plant_counts.get(p, 0) + 1
                if plant_counts:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    bars = ax2.barh(list(plant_counts.keys()),
                                   list(plant_counts.values()),
                                   color='#4CAF50')
                    ax2.set_xlabel("Count")
                    ax2.set_title("Plants Scanned")
                    for bar, val in zip(bars, plant_counts.values()):
                        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                                 str(val), va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

# ===== TAB 4 - HISTORY =====
with tab4:
    st.subheader("📋 Scan History")

    if not st.session_state.history:
        st.info("📸 No scan history yet. Start scanning leaves from the Detection tab!")
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.history      = []
            st.session_state.total_scans  = 0
            st.session_state.disease_count = 0
            st.session_state.healthy_count = 0
            st.rerun()

        for i, h in enumerate(reversed(st.session_state.history)):
            status_icon = "✅" if h['status'] == "Healthy" else "⚠️"
            with st.expander(f"{status_icon} [{h['date']} {h['time']}] {h['plant']} — {h['condition']}"):
                hcol1, hcol2, hcol3 = st.columns(3)
                with hcol1:
                    st.write(f"🌱 **Plant:** {h['plant']}")
                with hcol2:
                    st.write(f"🦠 **Condition:** {h['condition']}")
                with hcol3:
                    st.write(f"📊 **Confidence:** {h['confidence']}")

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "🌾 SmartAgriGuard | Major Project | Built with PyTorch, Streamlit & ❤️"
    "</p>",
    unsafe_allow_html=True
)