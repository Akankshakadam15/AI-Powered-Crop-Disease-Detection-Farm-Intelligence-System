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
import tempfile
import time

# ============ CONFIG ============
MODEL_PATH  = "plant_disease_model.pth"
NUM_CLASSES = 38
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenWeatherMap API Key
from dotenv import load_dotenv
import os
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "YOUR_API_KEY_HERE") 

# ============ MULTI-LANGUAGE SUPPORT ============
LANGUAGES = {
    "English": {
        "title": "🌾 SmartAgriGuard",
        "subtitle": "AI-Powered Crop Disease Detection & Farm Intelligence System",
        "upload_header": "📤 Upload Leaf Image",
        "results_header": "🔍 Analysis Results",
        "healthy_msg": "✅ Plant is Healthy!",
        "disease_msg": "⚠️ Disease Detected!",
        "plant_label": "🌱 Plant",
        "disease_label": "🦠 Disease",
        "confidence_label": "📊 Confidence",
        "severity_label": "⚡ Severity",
        "about_label": "📋 About",
        "treatment_label": "💊 Treatment",
        "top3_label": "📊 Top 3 Predictions",
        "gradcam_label": "🔥 Grad-CAM Heatmap (AI Focus Area)",
        "original_label": "Original",
        "ai_focus_label": "AI Focus (Red = High Attention)",
        "gradcam_caption": "🔴 Red areas = Where AI detected disease patterns",
        "tab_detection": "🔬 Disease Detection",
        "tab_weather": "🌦️ Weather & Risk",
        "tab_soil": "🌱 Soil Analysis",
        "tab_camera": "📷 Live Camera",
        "tab_video": "🎬 Video Detection",
        "tab_dashboard": "📊 Dashboard",
        "tab_history": "📋 Scan History",
        "tab_doctor": "🩺 AI Crop Doctor",
        "tab_smart_rec": "💡 Smart Recommendations",
        "tab_weather_pred": "🔮 Disease Forecast",
        "no_image_msg": "👈 Upload a leaf image to get started!",
        "how_to_use": "**How to use:**\n1. 📸 Click 'Browse files' and upload a leaf image\n2. 🔬 AI will detect disease automatically\n3. 💊 Get treatment recommendations\n4. 🔥 View Grad-CAM heatmap (where AI is looking)",
        "session_stats": "📊 Session Stats",
        "location_label": "📍 Your Location",
        "city_input": "City Name",
        "supported_plants": "ℹ️ Supported Plants",
        "scans": "Scans",
        "diseased": "Diseased",
        "healthy": "Healthy",
        "analyzing": "🔬 Analyzing leaf with AI...",
        "model_missing": "❌ Model file not found! Place `plant_disease_model.pth` in project folder.",
        "prevention_label": "🛡️ Prevention Tips",
        "medicine_label": "💉 Medicine Recommendation",
        "fertilizer_label": "🌿 Fertilizer Advice",
        "soil_header": "🌱 Soil Analysis & Recommendations",
        "soil_ph": "Soil pH",
        "soil_moisture": "Soil Moisture (%)",
        "soil_nitrogen": "Nitrogen (N) Level",
        "soil_phosphorus": "Phosphorus (P) Level",
        "soil_potassium": "Potassium (K) Level",
        "soil_type": "Soil Type",
        "analyze_soil_btn": "🔬 Analyze Soil",
        "camera_header": "📷 Live Camera Detection",
        "camera_start": "▶️ Start Camera",
        "camera_stop": "⏹️ Stop Camera",
        "video_header": "🎬 Video Frame Detection",
        "upload_video": "Upload a Video File",
        "process_video": "🎬 Process Video",
        "clear_history": "🗑️ Clear History",
        "no_history": "📸 No scan history yet. Start scanning leaves from the Detection tab!",
        "no_scans": "📸 No scans yet! Go to Disease Detection tab and scan some leaves.",
        "health_rate": "Health Rate",
        "total_scans": "Total Scans",
        "diseases_found": "Diseases Found",
        "healthy_plants": "Healthy Plants",
        "show_gradcam": "🔥 Show Grad-CAM Heatmap",
        "gradcam_help": "Highlights which part of the leaf the AI is focusing on",
        "image_loaded": "✅ Image loaded",
        "uploaded_leaf": "📸 Uploaded Leaf Image",
        "demo_mode": "Demo Mode - Sample Risk Analysis",
        "weather_header": "🌦️ Weather & Disease Risk Analysis",
        "risk_header": "🎯 Disease Risk Assessment",
        "farming_tips": "💡 Today's Farming Recommendations",
        "dashboard_header": "📊 Farm Analytics Dashboard",
        "history_header": "📋 Scan History",
    },
    "मराठी (Marathi)": {
        "title": "🌾 स्मार्टअॅग्रीगार्ड",
        "subtitle": "AI-आधारित पीक रोग ओळख आणि शेती बुद्धिमत्ता प्रणाली",
        "upload_header": "📤 पान प्रतिमा अपलोड करा",
        "results_header": "🔍 विश्लेषण निकाल",
        "healthy_msg": "✅ झाड निरोगी आहे!",
        "disease_msg": "⚠️ रोग आढळला!",
        "plant_label": "🌱 झाड",
        "disease_label": "🦠 रोग",
        "confidence_label": "📊 खात्री",
        "severity_label": "⚡ तीव्रता",
        "about_label": "📋 माहिती",
        "treatment_label": "💊 उपचार",
        "top3_label": "📊 शीर्ष 3 अंदाज",
        "gradcam_label": "🔥 ग्रॅड-CAM हीटमॅप (AI लक्ष क्षेत्र)",
        "original_label": "मूळ",
        "ai_focus_label": "AI लक्ष (लाल = जास्त लक्ष)",
        "gradcam_caption": "🔴 लाल भाग = जेथे AI ने रोगाचे नमुने शोधले",
        "tab_detection": "🔬 रोग ओळख",
        "tab_weather": "🌦️ हवामान आणि धोका",
        "tab_soil": "🌱 माती विश्लेषण",
        "tab_camera": "📷 थेट कॅमेरा",
        "tab_video": "🎬 व्हिडिओ ओळख",
        "tab_dashboard": "📊 डॅशबोर्ड",
        "tab_history": "📋 स्कॅन इतिहास",
        "tab_doctor": "🩺 AI पीक डॉक्टर",
        "tab_smart_rec": "💡 स्मार्ट शिफारसी",
        "tab_weather_pred": "🔮 रोग अंदाज",
        "no_image_msg": "👈 सुरुवात करण्यासाठी पानाची प्रतिमा अपलोड करा!",
        "how_to_use": "**कसे वापरावे:**\n1. 📸 'Browse files' क्लिक करा आणि पानाची प्रतिमा अपलोड करा\n2. 🔬 AI आपोआप रोग शोधेल\n3. 💊 उपचाराच्या शिफारसी मिळवा\n4. 🔥 ग्रॅड-CAM हीटमॅप पहा",
        "session_stats": "📊 सत्र सांख्यिकी",
        "location_label": "📍 तुमचे स्थान",
        "city_input": "शहराचे नाव",
        "supported_plants": "ℹ️ समर्थित झाडे",
        "scans": "स्कॅन्स",
        "diseased": "आजारी",
        "healthy": "निरोगी",
        "analyzing": "🔬 AI ने पान विश्लेषण करत आहे...",
        "model_missing": "❌ मॉडेल फाइल सापडली नाही! `plant_disease_model.pth` प्रोजेक्ट फोल्डरमध्ये ठेवा.",
        "prevention_label": "🛡️ प्रतिबंध टिप्स",
        "medicine_label": "💉 औषध शिफारस",
        "fertilizer_label": "🌿 खत सल्ला",
        "soil_header": "🌱 माती विश्लेषण आणि शिफारसी",
        "soil_ph": "माती pH",
        "soil_moisture": "माती ओलावा (%)",
        "soil_nitrogen": "नायट्रोजन (N) पातळी",
        "soil_phosphorus": "फॉस्फरस (P) पातळी",
        "soil_potassium": "पोटॅशियम (K) पातळी",
        "soil_type": "मातीचा प्रकार",
        "analyze_soil_btn": "🔬 माती विश्लेषण करा",
        "camera_header": "📷 थेट कॅमेरा ओळख",
        "camera_start": "▶️ कॅमेरा सुरू करा",
        "camera_stop": "⏹️ कॅमेरा थांबवा",
        "video_header": "🎬 व्हिडिओ फ्रेम ओळख",
        "upload_video": "व्हिडिओ फाइल अपलोड करा",
        "process_video": "🎬 व्हिडिओ प्रक्रिया करा",
        "clear_history": "🗑️ इतिहास साफ करा",
        "no_history": "📸 अद्याप कोणताही स्कॅन इतिहास नाही. ओळख टॅबमधून पाने स्कॅन करणे सुरू करा!",
        "no_scans": "📸 अद्याप कोणतेही स्कॅन नाहीत! रोग ओळख टॅबवर जा आणि पाने स्कॅन करा.",
        "health_rate": "आरोग्य दर",
        "total_scans": "एकूण स्कॅन्स",
        "diseases_found": "रोग आढळले",
        "healthy_plants": "निरोगी झाडे",
        "show_gradcam": "🔥 ग्रॅड-CAM हीटमॅप दाखवा",
        "gradcam_help": "AI पानाच्या कोणत्या भागावर लक्ष केंद्रित करत आहे ते दाखवते",
        "image_loaded": "✅ प्रतिमा लोड झाली",
        "uploaded_leaf": "📸 अपलोड केलेली पानाची प्रतिमा",
        "demo_mode": "डेमो मोड - नमुना जोखीम विश्लेषण",
        "weather_header": "🌦️ हवामान आणि रोग जोखीम विश्लेषण",
        "risk_header": "🎯 रोग जोखीम मूल्यांकन",
        "farming_tips": "💡 आजच्या शेतीच्या शिफारसी",
        "dashboard_header": "📊 शेत विश्लेषण डॅशबोर्ड",
        "history_header": "📋 स्कॅन इतिहास",
    },
    "हिंदी (Hindi)": {
        "title": "🌾 स्मार्टएग्रीगार्ड",
        "subtitle": "AI-संचालित फसल रोग पहचान और कृषि बुद्धिमत्ता प्रणाली",
        "upload_header": "📤 पत्ती की छवि अपलोड करें",
        "results_header": "🔍 विश्लेषण परिणाम",
        "healthy_msg": "✅ पौधा स्वस्थ है!",
        "disease_msg": "⚠️ रोग पाया गया!",
        "plant_label": "🌱 पौधा",
        "disease_label": "🦠 रोग",
        "confidence_label": "📊 विश्वास",
        "severity_label": "⚡ गंभीरता",
        "about_label": "📋 जानकारी",
        "treatment_label": "💊 उपचार",
        "top3_label": "📊 शीर्ष 3 पूर्वानुमान",
        "gradcam_label": "🔥 ग्रेड-CAM हीटमैप (AI फोकस क्षेत्र)",
        "original_label": "मूल",
        "ai_focus_label": "AI फोकस (लाल = उच्च ध्यान)",
        "gradcam_caption": "🔴 लाल क्षेत्र = जहाँ AI ने रोग के पैटर्न पाए",
        "tab_detection": "🔬 रोग पहचान",
        "tab_weather": "🌦️ मौसम और जोखिम",
        "tab_soil": "🌱 मिट्टी विश्लेषण",
        "tab_camera": "📷 लाइव कैमरा",
        "tab_video": "🎬 वीडियो पहचान",
        "tab_dashboard": "📊 डैशबोर्ड",
        "tab_history": "📋 स्कैन इतिहास",
        "tab_doctor": "🩺 AI फसल डॉक्टर",
        "tab_smart_rec": "💡 स्मार्ट सिफारिशें",
        "tab_weather_pred": "🔮 रोग पूर्वानुमान",
        "no_image_msg": "👈 शुरू करने के लिए एक पत्ती की छवि अपलोड करें!",
        "how_to_use": "**उपयोग कैसे करें:**\n1. 📸 'Browse files' क्लिक करें और पत्ती की छवि अपलोड करें\n2. 🔬 AI स्वतः रोग का पता लगाएगा\n3. 💊 उपचार की सिफारिशें प्राप्त करें\n4. 🔥 Grad-CAM हीटमैप देखें",
        "session_stats": "📊 सत्र आँकड़े",
        "location_label": "📍 आपका स्थान",
        "city_input": "शहर का नाम",
        "supported_plants": "ℹ️ समर्थित पौधे",
        "scans": "स्कैन",
        "diseased": "रोगग्रस्त",
        "healthy": "स्वस्थ",
        "analyzing": "🔬 AI से पत्ती का विश्लेषण हो रहा है...",
        "model_missing": "❌ मॉडल फ़ाइल नहीं मिली! `plant_disease_model.pth` को प्रोजेक्ट फ़ोल्डर में रखें।",
        "prevention_label": "🛡️ रोकथाम के उपाय",
        "medicine_label": "💉 दवा की सिफारिश",
        "fertilizer_label": "🌿 उर्वरक सलाह",
        "soil_header": "🌱 मिट्टी विश्लेषण और सिफारिशें",
        "soil_ph": "मिट्टी pH",
        "soil_moisture": "मिट्टी नमी (%)",
        "soil_nitrogen": "नाइट्रोजन (N) स्तर",
        "soil_phosphorus": "फॉस्फोरस (P) स्तर",
        "soil_potassium": "पोटेशियम (K) स्तर",
        "soil_type": "मिट्टी का प्रकार",
        "analyze_soil_btn": "🔬 मिट्टी विश्लेषण करें",
        "camera_header": "📷 लाइव कैमरा पहचान",
        "camera_start": "▶️ कैमरा शुरू करें",
        "camera_stop": "⏹️ कैमरा बंद करें",
        "video_header": "🎬 वीडियो फ्रेम पहचान",
        "upload_video": "वीडियो फ़ाइल अपलोड करें",
        "process_video": "🎬 वीडियो प्रोसेस करें",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "no_history": "📸 अभी तक कोई स्कैन इतिहास नहीं। पहचान टैब से पत्तियां स्कैन करना शुरू करें!",
        "no_scans": "📸 अभी तक कोई स्कैन नहीं! रोग पहचान टैब पर जाएं और पत्तियां स्कैन करें।",
        "health_rate": "स्वास्थ्य दर",
        "total_scans": "कुल स्कैन",
        "diseases_found": "रोग मिले",
        "healthy_plants": "स्वस्थ पौधे",
        "show_gradcam": "🔥 ग्रेड-CAM हीटमैप दिखाएं",
        "gradcam_help": "AI पत्ती के किस हिस्से पर ध्यान दे रहा है वह दिखाता है",
        "image_loaded": "✅ छवि लोड हुई",
        "uploaded_leaf": "📸 अपलोड की गई पत्ती की छवि",
        "demo_mode": "डेमो मोड - नमूना जोखिम विश्लेषण",
        "weather_header": "🌦️ मौसम और रोग जोखिम विश्लेषण",
        "risk_header": "🎯 रोग जोखिम मूल्यांकन",
        "farming_tips": "💡 आज की कृषि सिफारिशें",
        "dashboard_header": "📊 खेत विश्लेषण डैशबोर्ड",
        "history_header": "📋 स्कैन इतिहास",
    }
}

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

# ============ DISEASE INFO (Extended with prevention + medicine + fertilizer) ============
DISEASE_INFO = {
    "Apple_scab": (
        "Fungal disease causing dark scabs on leaves and fruit.",
        "🌿 Use fungicide sprays. Remove infected leaves. Avoid overhead watering.",
        "🛡️ Plant resistant apple varieties. Apply dormant oil sprays. Rake and destroy fallen leaves.",
        "💉 Captan 50WP (2g/L), Mancozeb 75WP (2.5g/L), or Myclobutanil 10WP (1g/L).",
        "🌿 Apply balanced NPK 10-10-10 fertilizer. Avoid excess nitrogen which promotes soft tissue susceptible to scab."
    ),
    "Black_rot": (
        "Fungal disease causing black rotting of fruit and leaves.",
        "✂️ Prune infected branches. Apply copper-based fungicide. Destroy fallen fruit.",
        "🛡️ Remove mummified fruit. Prune dead wood. Ensure good canopy airflow.",
        "💉 Copper Oxychloride 50WP (3g/L), Thiophanate-methyl 70WP (1g/L).",
        "🌿 Balanced K2O fertilizer improves fruit skin strength. Avoid excess nitrogen."
    ),
    "Cedar_apple_rust": (
        "Fungal disease causing orange spots on leaves.",
        "🍎 Remove nearby cedar trees. Use myclobutanil fungicide in spring.",
        "🛡️ Plant rust-resistant varieties. Apply fungicide from pink bud stage every 7-10 days.",
        "💉 Myclobutanil (Nova) 1g/L, Propiconazole 25EC (1ml/L) applied at pink bud stage.",
        "🌿 Potassium-rich fertilizer (0-0-60 or SOP) improves resistance to fungal diseases."
    ),
    "Powdery_mildew": (
        "Fungal disease causing white powdery coating on leaves.",
        "💨 Improve air circulation. Apply sulfur-based fungicide. Avoid wet leaves.",
        "🛡️ Space plants properly. Avoid overhead irrigation. Remove infected shoot tips early.",
        "💉 Sulfur 80WP (3g/L), Trifloxystrobin 25SC (0.5ml/L), or Potassium bicarbonate solution.",
        "🌿 Avoid high nitrogen fertilizers. Apply calcium (CaNO3) to strengthen cell walls."
    ),
    "Cercospora_leaf_spot": (
        "Fungal disease causing gray spots on leaves.",
        "🌾 Rotate crops. Apply chlorothalonil fungicide. Remove crop debris.",
        "🛡️ Use certified disease-free seeds. Maintain proper plant spacing. Crop rotation every 2-3 years.",
        "💉 Chlorothalonil 75WP (2g/L), Azoxystrobin 23SC (1ml/L), Propiconazole 25EC (1ml/L).",
        "🌿 Apply micronutrient mix with zinc and manganese. Avoid excess nitrogen after disease onset."
    ),
    "Common_rust": (
        "Fungal disease causing rust-colored pustules on leaves.",
        "🌽 Plant resistant varieties. Apply fungicide early. Monitor regularly.",
        "🛡️ Grow resistant hybrids. Avoid late planting. Monitor from V4-V6 stage.",
        "💉 Propiconazole 25EC (1ml/L), Tebuconazole 250EC (1ml/L), Azoxystrobin + Propiconazole.",
        "🌿 Potassium silicate foliar spray improves leaf strength. Ensure adequate P and K nutrition."
    ),
    "Northern_Leaf_Blight": (
        "Fungal disease causing large tan lesions on leaves.",
        "🌿 Use resistant hybrids. Apply triazole fungicide at early stages.",
        "🛡️ Rotate with soybean or wheat. Bury crop residues. Use resistant hybrids.",
        "💉 Propiconazole 25EC (1ml/L), Azoxystrobin 23SC (1ml/L), Pyraclostrobin 20WG (1g/L).",
        "🌿 Balanced NPK nutrition. Adequate zinc supplementation for better plant immunity."
    ),
    "Bacterial_spot": (
        "Bacterial disease causing small dark spots on leaves.",
        "🦠 Use copper bactericide. Avoid overhead irrigation. Remove infected plant parts.",
        "🛡️ Use disease-free transplants. Avoid working in wet fields. Copper sprays preventively.",
        "💉 Copper hydroxide 77WP (3g/L), Streptomycin sulfate 90SP (0.5g/L), Kasugamycin 3SL (1.5ml/L).",
        "🌿 Calcium nitrate foliar spray strengthens cell walls. Avoid excess nitrogen."
    ),
    "Early_blight": (
        "Fungal disease causing dark spots with rings on leaves.",
        "🍅 Apply mancozeb fungicide. Mulch soil. Water at base of plant.",
        "🛡️ Rotate crops with non-solanaceous crops. Remove lower leaves touching soil. Mulch well.",
        "💉 Mancozeb 75WP (2.5g/L), Chlorothalonil 75WP (2g/L), Difenoconazole 25EC (0.5ml/L).",
        "🌿 Balanced NPK. Adequate potassium reduces disease susceptibility. Calcium supplements reduce tissue damage."
    ),
    "Late_blight": (
        "Serious disease causing water-soaked lesions on leaves and stems.",
        "⚠️ Apply metalaxyl fungicide immediately. Remove infected plants. Improve drainage.",
        "🛡️ URGENT: Monitor daily. Destroy all infected plant material. Avoid overhead irrigation. Use resistant varieties.",
        "💉 Metalaxyl + Mancozeb 72WP (2.5g/L), Cymoxanil 8% + Mancozeb 64WP (2.5g/L), Dimethomorph 50WP (1g/L).",
        "🌿 Phosphorus-rich fertilizer (DAP) strengthens roots. Avoid excess nitrogen. Apply potassium to harden tissues."
    ),
    "Leaf_Mold": (
        "Fungal disease causing yellow spots on upper leaf surface.",
        "🍃 Reduce humidity. Improve ventilation. Apply chlorothalonil fungicide.",
        "🛡️ Use resistant varieties. Reduce humidity in greenhouse. Avoid dense planting.",
        "💉 Chlorothalonil 75WP (2g/L), Copper oxychloride 50WP (3g/L), Iprodione 50WP (1.5g/L).",
        "🌿 Balanced fertilization. Excess nitrogen worsens leaf mold. Apply calcium foliar spray."
    ),
    "Septoria_leaf_spot": (
        "Fungal disease causing small circular spots with dark borders.",
        "🌿 Remove lower infected leaves. Apply fungicide. Avoid wetting foliage.",
        "🛡️ Crop rotation 2 years minimum. Stake plants for air circulation. Prune basal leaves.",
        "💉 Chlorothalonil 75WP (2g/L), Mancozeb 75WP (2.5g/L), Thiophanate-methyl 70WP (1g/L).",
        "🌿 Avoid high nitrogen. Calcium foliar sprays improve resistance. Mulch to prevent soil splash."
    ),
    "Spider_mites": (
        "Pest causing stippling and webbing on leaves.",
        "🕷️ Apply miticide or neem oil. Increase humidity. Introduce predatory mites.",
        "🛡️ Regular scouting from underside of leaves. Maintain plant hydration. Avoid drought stress.",
        "💉 Abamectin 1.8EC (1ml/L), Spiromesifen 22.9SC (1ml/L), Neem oil 5% spray, Bifenazate 43SC (1ml/L).",
        "🌿 Silicon fertilizer (potassium silicate) strengthens leaf epidermis against mite feeding."
    ),
    "Target_Spot": (
        "Fungal disease causing target-like circular spots on leaves.",
        "🎯 Apply azoxystrobin fungicide. Rotate crops. Remove infected debris.",
        "🛡️ Destroy crop debris. Rotate with cereals. Avoid dense canopy.",
        "💉 Azoxystrobin 23SC (1ml/L), Fluxapyroxad + Pyraclostrobin (0.8ml/L), Boscalid 50WG (0.6g/L).",
        "🌿 Balanced fertilizer. Potassium improves overall disease resistance."
    ),
    "Yellow_Leaf_Curl_Virus": (
        "Viral disease causing yellowing and curling of leaves.",
        "🦟 Control whitefly vectors. Use reflective mulches. Remove infected plants.",
        "🛡️ Use sticky yellow traps. Reflective silver mulch repels whiteflies. Plant barrier crops.",
        "💉 No direct cure. Control vector: Imidacloprid 17.8SL (0.5ml/L) or Thiamethoxam 25WG (0.3g/L) for whiteflies.",
        "🌿 Balanced nutrition. Avoid excess nitrogen which attracts whiteflies. Silicon supplements improve resistance."
    ),
    "mosaic_virus": (
        "Viral disease causing mosaic pattern on leaves.",
        "🧬 No cure. Remove infected plants. Control aphid vectors. Use virus-free seeds.",
        "🛡️ Use certified virus-free seeds/transplants. Control aphid populations. Disinfect tools.",
        "💉 No direct cure. Control aphids: Dimethoate 30EC (1ml/L), Imidacloprid 17.8SL (0.5ml/L).",
        "🌿 Avoid excess nitrogen. Balanced nutrition. Silica sprays strengthen plant against virus spread."
    ),
    "Haunglongbing": (
        "Serious citrus disease spread by psyllid insects.",
        "🍊 Remove infected trees. Control psyllid population. Use certified disease-free plants.",
        "🛡️ CRITICAL: No cure exists. Monitor for Asian citrus psyllid. Remove infected trees promptly.",
        "💉 No cure. Control psyllid vector: Imidacloprid 17.8SL soil drench, Dimethoate 30EC foliar spray.",
        "🌿 Foliar micronutrient sprays (zinc, manganese, boron) delay symptom progression in early stages."
    ),
    "Esca": (
        "Fungal complex causing internal wood decay in grapevines.",
        "🍇 Prune infected wood. Apply wound protectants. Avoid pruning in wet weather.",
        "🛡️ Prune during dry weather. Apply pruning wound paste (Trichoderma-based). Remove infected canes.",
        "💉 Sodium arsenite (where legal) historically used; currently: Tebuconazole wound treatment, Trichoderma harzianum bio-fungicide.",
        "🌿 Balanced vine nutrition. Adequate boron and zinc foliar spray. Avoid excess nitrogen."
    ),
    "Leaf_scorch": (
        "Fungal disease causing browning and scorching of leaf margins.",
        "🍓 Apply appropriate fungicide. Ensure proper irrigation. Remove infected leaves.",
        "🛡️ Remove infected leaves promptly. Avoid overhead irrigation. Ensure good plant spacing.",
        "💉 Captan 50WP (2g/L), Copper oxychloride 50WP (3g/L), Myclobutanil 10WP (1g/L).",
        "🌿 Calcium foliar spray reduces tip burn. Balanced K fertilizer. Avoid drought stress."
    ),
    "healthy": (
        "Plant appears healthy with no visible signs of disease.",
        "✅ Continue regular monitoring. Maintain proper irrigation and fertilization.",
        "🛡️ Preventive copper or sulfur spray monthly. Rotate crops. Monitor weekly for early symptoms.",
        "💉 No treatment needed. Preventive: Bordeaux mixture (1%) monthly spray as prophylactic.",
        "🌿 Maintain balanced NPK fertilization schedule. Soil test annually. Apply micronutrients as needed."
    ),
}

# ============ NEW FEATURE 1: AI CROP DOCTOR KNOWLEDGE BASE ============
# Fully offline — no internet needed. Pure Python dict lookup.
CROP_DOCTOR_KB = {
    # ---- FUNGAL ----
    "late blight": {
        "symptoms": "Water-soaked dark lesions on leaves/stems, white fungal growth on undersides, rapid plant collapse in humid weather.",
        "chemical": [
            {"name": "Metalaxyl + Mancozeb 72WP", "dose": "2.5 g/L water", "interval": "Every 7 days"},
            {"name": "Cymoxanil 8% + Mancozeb 64WP", "dose": "2.5 g/L water", "interval": "Every 7-10 days"},
            {"name": "Dimethomorph 50WP", "dose": "1 g/L water", "interval": "Every 10 days"},
        ],
        "organic": [
            "🌿 Bordeaux mixture (1%) spray every 7 days",
            "🌿 Copper sulfate + lime solution (0.5%)",
            "🌿 Trichoderma viride bio-fungicide (5g/L)",
            "🌿 Remove & burn all infected plant parts immediately",
        ],
        "severity": "HIGH — Act within 24 hours!",
        "shop_hint": "Available at: Krishi Seva Kendra, local agri-input shops. Ask for 'Ridomil Gold' or 'Mancozeb'.",
    },
    "early blight": {
        "symptoms": "Dark brown spots with concentric rings (target-board pattern) on older leaves first.",
        "chemical": [
            {"name": "Mancozeb 75WP", "dose": "2.5 g/L water", "interval": "Every 7-10 days"},
            {"name": "Chlorothalonil 75WP", "dose": "2 g/L water", "interval": "Every 7 days"},
            {"name": "Difenoconazole 25EC", "dose": "0.5 ml/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Neem oil spray 5ml/L every 10 days",
            "🌿 Baking soda solution (5g/L) as foliar spray",
            "🌿 Remove lower infected leaves and mulch soil",
            "🌿 Garlic extract spray (50g garlic in 1L water)",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Dithane M-45' (Mancozeb) or 'Kavach' (Chlorothalonil) at agri shops.",
    },
    "powdery mildew": {
        "symptoms": "White powdery patches on leaf surface, distorted growth, yellowing.",
        "chemical": [
            {"name": "Sulfur 80WP", "dose": "3 g/L water", "interval": "Every 7 days"},
            {"name": "Trifloxystrobin 25SC", "dose": "0.5 ml/L water", "interval": "Every 14 days"},
            {"name": "Myclobutanil 10WP", "dose": "1 g/L water", "interval": "Every 10-14 days"},
        ],
        "organic": [
            "🌿 Potassium bicarbonate (5g/L) spray",
            "🌿 Milk spray (40% milk + 60% water) weekly",
            "🌿 Neem oil 5ml/L + 2 drops dish soap",
            "🌿 Improve air circulation by pruning dense canopy",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Sulfex' (Sulfur) or 'Flint' (Trifloxystrobin). Available at most agri stores.",
    },
    "bacterial spot": {
        "symptoms": "Small water-soaked spots turning dark with yellow halo, spots may fall out leaving holes.",
        "chemical": [
            {"name": "Copper hydroxide 77WP", "dose": "3 g/L water", "interval": "Every 7 days"},
            {"name": "Streptomycin sulfate 90SP", "dose": "0.5 g/L water", "interval": "Every 5-7 days"},
            {"name": "Kasugamycin 3SL", "dose": "1.5 ml/L water", "interval": "Every 7 days"},
        ],
        "organic": [
            "🌿 Copper sulfate Bordeaux mixture (1%) spray",
            "🌿 Avoid overhead irrigation completely",
            "🌿 Remove infected leaves and bury/burn them",
            "🌿 Pseudomonas fluorescens bio-agent spray (10g/L)",
        ],
        "severity": "MEDIUM-HIGH",
        "shop_hint": "Ask for 'Kocide' (Copper hydroxide) or 'Agrimycin' (Streptomycin) at agri shops.",
    },
    "apple scab": {
        "symptoms": "Olive-green to dark scabs on leaves/fruit, deformed fruit, premature leaf drop.",
        "chemical": [
            {"name": "Captan 50WP", "dose": "2 g/L water", "interval": "Every 7-10 days"},
            {"name": "Mancozeb 75WP", "dose": "2.5 g/L water", "interval": "Every 7 days"},
            {"name": "Myclobutanil 10WP", "dose": "1 g/L water", "interval": "Every 10-14 days"},
        ],
        "organic": [
            "🌿 Lime sulfur spray during dormant season",
            "🌿 Neem oil spray 5ml/L every 10 days",
            "🌿 Rake and destroy all fallen leaves",
            "🌿 Plant resistant apple varieties next season",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Captaf' (Captan) or 'Bavistin' at your nearest nursery or agri-input shop.",
    },
    "black rot": {
        "symptoms": "Circular black lesions on fruit, 'frog-eye' leaf spots, mummified fruits remain on tree.",
        "chemical": [
            {"name": "Copper Oxychloride 50WP", "dose": "3 g/L water", "interval": "Every 7-10 days"},
            {"name": "Thiophanate-methyl 70WP", "dose": "1 g/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Prune and destroy all mummified fruits",
            "🌿 Bordeaux mixture 1% spray",
            "🌿 Improve canopy ventilation",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Blitox' (Copper Oxychloride) or 'Topsin-M' (Thiophanate-methyl).",
    },
    "leaf mold": {
        "symptoms": "Yellow spots on upper leaf, olive-green/brown fuzzy mold on underside.",
        "chemical": [
            {"name": "Chlorothalonil 75WP", "dose": "2 g/L water", "interval": "Every 7 days"},
            {"name": "Copper oxychloride 50WP", "dose": "3 g/L water", "interval": "Every 7-10 days"},
            {"name": "Iprodione 50WP", "dose": "1.5 g/L water", "interval": "Every 10 days"},
        ],
        "organic": [
            "🌿 Reduce greenhouse/tunnel humidity below 85%",
            "🌿 Increase plant spacing for airflow",
            "🌿 Trichoderma harzianum spray 5g/L",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Kavach' or 'Blitox' at agri shops.",
    },
    "septoria leaf spot": {
        "symptoms": "Small circular spots with dark border and tan/gray center, tiny black dots inside spots.",
        "chemical": [
            {"name": "Chlorothalonil 75WP", "dose": "2 g/L water", "interval": "Every 7 days"},
            {"name": "Mancozeb 75WP", "dose": "2.5 g/L water", "interval": "Every 7 days"},
            {"name": "Thiophanate-methyl 70WP", "dose": "1 g/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Remove and destroy infected lower leaves",
            "🌿 Mulch soil to prevent spore splash",
            "🌿 Neem oil spray 5ml/L every 10 days",
            "🌿 Copper soap spray (organic)",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Dithane M-45' (Mancozeb) or 'Topsin' at agri shops.",
    },
    "spider mites": {
        "symptoms": "Tiny yellow/white stippling on leaves, fine webbing on underside, bronze/rusty leaf color.",
        "chemical": [
            {"name": "Abamectin 1.8EC", "dose": "1 ml/L water", "interval": "Every 7 days (2 sprays)"},
            {"name": "Spiromesifen 22.9SC", "dose": "1 ml/L water", "interval": "Every 10-14 days"},
            {"name": "Bifenazate 43SC", "dose": "1 ml/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Neem oil 5ml/L + soap spray weekly",
            "🌿 Increase humidity (mites hate moisture)",
            "🌿 Release predatory mites (Phytoseiulus persimilis)",
            "🌿 Garlic + chili extract spray",
            "🌿 Strong water spray to dislodge mites",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Oberon' (Spiromesifen) or 'Vertimec' (Abamectin) at agri shops.",
    },
    "target spot": {
        "symptoms": "Brown circular target-like spots with concentric rings on leaves.",
        "chemical": [
            {"name": "Azoxystrobin 23SC", "dose": "1 ml/L water", "interval": "Every 10-14 days"},
            {"name": "Fluxapyroxad + Pyraclostrobin", "dose": "0.8 ml/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Crop rotation with non-solanaceous crops",
            "🌿 Remove crop debris after harvest",
            "🌿 Trichoderma viride soil application",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Amistar' (Azoxystrobin) at agri shops.",
    },
    "yellow leaf curl virus": {
        "symptoms": "Upward leaf curling, yellowing of leaf margins, stunted growth, smaller leaves.",
        "chemical": [
            {"name": "Imidacloprid 17.8SL (for whitefly control)", "dose": "0.5 ml/L water", "interval": "Every 7-10 days"},
            {"name": "Thiamethoxam 25WG (for whitefly control)", "dose": "0.3 g/L water", "interval": "Every 10 days"},
        ],
        "organic": [
            "🌿 Yellow sticky traps (40/acre) to catch whiteflies",
            "🌿 Reflective silver/aluminum mulch repels whiteflies",
            "🌿 Neem oil 5ml/L spray for whitefly control",
            "🌿 Remove & destroy infected plants immediately",
            "🌿 Plant marigold as border crop (whitefly repellent)",
        ],
        "severity": "HIGH — No cure. Prevent spread!",
        "shop_hint": "Ask for 'Confidor' (Imidacloprid) or 'Actara' (Thiamethoxam). Yellow sticky traps at agri shops.",
    },
    "mosaic virus": {
        "symptoms": "Mosaic pattern of light/dark green on leaves, leaf distortion, stunted growth.",
        "chemical": [
            {"name": "Dimethoate 30EC (aphid control)", "dose": "1 ml/L water", "interval": "Every 7 days"},
            {"name": "Imidacloprid 17.8SL (aphid control)", "dose": "0.5 ml/L water", "interval": "Every 7-10 days"},
        ],
        "organic": [
            "🌿 No direct cure — focus on vector (aphid) control",
            "🌿 Neem oil spray for aphids",
            "🌿 Use virus-free certified seeds",
            "🌿 Remove infected plants immediately",
            "🌿 Disinfect tools with bleach solution",
        ],
        "severity": "HIGH — No cure. Remove infected plants!",
        "shop_hint": "Ask for 'Rogor' (Dimethoate) for aphid control. Use virus-free seeds from certified sources.",
    },
    "cercospora leaf spot": {
        "symptoms": "Gray to tan circular spots with reddish-purple border on leaves.",
        "chemical": [
            {"name": "Chlorothalonil 75WP", "dose": "2 g/L water", "interval": "Every 7-10 days"},
            {"name": "Azoxystrobin 23SC", "dose": "1 ml/L water", "interval": "Every 14 days"},
            {"name": "Propiconazole 25EC", "dose": "1 ml/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Crop rotation every 2-3 years",
            "🌿 Remove and burn infected leaves",
            "🌿 Trichoderma harzianum soil treatment",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Kavach' or 'Tilt' (Propiconazole) at agri shops.",
    },
    "common rust": {
        "symptoms": "Orange-brown powdery pustules on both leaf surfaces, leaves may yellow and die.",
        "chemical": [
            {"name": "Propiconazole 25EC", "dose": "1 ml/L water", "interval": "Every 14 days"},
            {"name": "Tebuconazole 250EC", "dose": "1 ml/L water", "interval": "Every 14 days"},
            {"name": "Azoxystrobin 23SC", "dose": "1 ml/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Plant rust-resistant corn hybrids",
            "🌿 Sulfur dust application early in season",
            "🌿 Monitor from V4-V6 growth stage",
        ],
        "severity": "MEDIUM",
        "shop_hint": "Ask for 'Tilt' (Propiconazole) or 'Folicur' (Tebuconazole) at agri shops.",
    },
    "northern leaf blight": {
        "symptoms": "Large cigar-shaped tan lesions (2-15 cm) on leaves, gray-green water-soaked initially.",
        "chemical": [
            {"name": "Propiconazole 25EC", "dose": "1 ml/L water", "interval": "Every 14 days"},
            {"name": "Azoxystrobin 23SC", "dose": "1 ml/L water", "interval": "Every 14 days"},
            {"name": "Pyraclostrobin 20WG", "dose": "1 g/L water", "interval": "Every 14 days"},
        ],
        "organic": [
            "🌿 Use resistant hybrids — most effective strategy",
            "🌿 Bury or remove crop residues post-harvest",
            "🌿 Rotate with soybean or wheat",
        ],
        "severity": "MEDIUM-HIGH",
        "shop_hint": "Ask for 'Tilt' or 'Amistar Top' at agri shops.",
    },
    "haunglongbing": {
        "symptoms": "Blotchy mottled yellowing (not uniform), small lopsided fruit, bitter taste, stunted growth.",
        "chemical": [
            {"name": "Imidacloprid 17.8SL soil drench (psyllid control)", "dose": "2 ml/L water", "interval": "Every 3 months"},
            {"name": "Dimethoate 30EC foliar (psyllid control)", "dose": "1 ml/L water", "interval": "Every 15 days"},
        ],
        "organic": [
            "🌿 CRITICAL: No cure exists for HLB",
            "🌿 Remove and destroy infected trees immediately",
            "🌿 Monitor for Asian citrus psyllid regularly",
            "🌿 Use certified disease-free nursery plants",
            "🌿 Yellow sticky traps around orchard border",
        ],
        "severity": "CRITICAL — No cure. Tree removal required!",
        "shop_hint": "Contact your local Horticulture Department or KVK for support.",
    },
    "esca": {
        "symptoms": "Tiger-stripe pattern on leaves (interveinal yellowing/reddening), wood shows brown streaks when cut.",
        "chemical": [
            {"name": "Tebuconazole wound treatment paste", "dose": "Apply neat to pruning wounds", "interval": "After each pruning"},
        ],
        "organic": [
            "🌿 Prune during dry weather only",
            "🌿 Apply Trichoderma harzianum paste on pruning wounds",
            "🌿 Remove and burn infected canes",
            "🌿 Disinfect pruning tools with bleach between cuts",
        ],
        "severity": "HIGH",
        "shop_hint": "Ask for Trichoderma bio-fungicide at agri shops. Consult viticulture extension officer.",
    },
    "leaf scorch": {
        "symptoms": "Browning and scorching of leaf margins and tips, dry papery texture.",
        "chemical": [
            {"name": "Captan 50WP", "dose": "2 g/L water", "interval": "Every 10 days"},
            {"name": "Copper oxychloride 50WP", "dose": "3 g/L water", "interval": "Every 7-10 days"},
        ],
        "organic": [
            "🌿 Remove and destroy infected leaves",
            "🌿 Avoid overhead irrigation",
            "🌿 Calcium foliar spray (CaNO3 1g/L)",
        ],
        "severity": "LOW-MEDIUM",
        "shop_hint": "Ask for 'Captaf' (Captan) or 'Blitox' (Copper Oxychloride).",
    },
    "healthy": {
        "symptoms": "No symptoms — plant looks good!",
        "chemical": [],
        "organic": [
            "🌿 Continue regular monitoring (weekly)",
            "🌿 Preventive Bordeaux mixture spray monthly",
            "🌿 Maintain proper irrigation schedule",
            "🌿 Balanced NPK fertilization",
        ],
        "severity": "NONE — Plant is healthy ✅",
        "shop_hint": "No treatment needed. Consider preventive bio-fungicide spray monthly.",
    },
}

# ============ NEW FEATURE 2: SMART RECOMMENDATION ENGINE ============
SMART_RECOMMENDATIONS = {
    # Maps last detected disease → smart action plan
    "Late_blight": {
        "urgency": "🚨 URGENT — Act Today!",
        "priority_action": "Spray Metalaxyl + Mancozeb 72WP (2.5g/L) within 24 hours. Remove visibly infected plants.",
        "pesticide_schedule": [
            {"day": "Day 0 (Today)", "action": "Spray Metalaxyl+Mancozeb 2.5g/L", "type": "Chemical"},
            {"day": "Day 7", "action": "Re-spray if humid. Switch to Cymoxanil+Mancozeb to avoid resistance", "type": "Chemical"},
            {"day": "Day 14", "action": "Apply Trichoderma viride as soil drench (preventive)", "type": "Organic"},
            {"day": "Day 21", "action": "Scout & monitor. Preventive copper spray if rain forecast", "type": "Preventive"},
        ],
        "organic_alternatives": "Bordeaux mixture (1%) every 7 days + remove infected plants + burn debris",
        "dose_reminder": "⚠️ Always mix in correct ratio. Overdosing causes leaf burn!",
        "harvest_interval": "⏰ Metalaxyl: 7-day PHI (Pre-Harvest Interval) — do not harvest within 7 days of last spray.",
    },
    "Early_blight": {
        "urgency": "⚠️ Act within 3-5 days",
        "priority_action": "Spray Mancozeb 75WP (2.5g/L) and remove lower infected leaves.",
        "pesticide_schedule": [
            {"day": "Day 0", "action": "Remove lower infected leaves. Spray Mancozeb 2.5g/L", "type": "Chemical"},
            {"day": "Day 7", "action": "Re-spray Mancozeb. Check soil moisture — mulch if needed", "type": "Chemical"},
            {"day": "Day 14", "action": "Switch to Difenoconazole 0.5ml/L for curative effect", "type": "Chemical"},
            {"day": "Day 21", "action": "Neem oil spray 5ml/L as organic follow-up", "type": "Organic"},
        ],
        "organic_alternatives": "Neem oil 5ml/L + baking soda 5g/L spray every 10 days",
        "dose_reminder": "⚠️ Mancozeb PHI: 3-5 days before harvest",
        "harvest_interval": "⏰ Mancozeb: 3-5 day PHI. Difenoconazole: 7-day PHI.",
    },
    "Powdery_mildew": {
        "urgency": "⚠️ Act within 5-7 days",
        "priority_action": "Spray Sulfur 80WP (3g/L) in evening. Improve ventilation.",
        "pesticide_schedule": [
            {"day": "Day 0", "action": "Spray Sulfur 80WP 3g/L (evening only — not in hot sun!)", "type": "Chemical"},
            {"day": "Day 7", "action": "Re-spray Sulfur. Prune dense canopy for airflow", "type": "Chemical"},
            {"day": "Day 14", "action": "Switch to Trifloxystrobin 0.5ml/L for systemic action", "type": "Chemical"},
            {"day": "Day 21", "action": "Potassium bicarbonate spray (5g/L) as organic option", "type": "Organic"},
        ],
        "organic_alternatives": "Milk spray (40% milk+60% water) weekly + potassium bicarbonate 5g/L",
        "dose_reminder": "⚠️ Do NOT apply sulfur when temp > 35°C — causes leaf burn!",
        "harvest_interval": "⏰ Sulfur: 1-day PHI. Trifloxystrobin: 3-day PHI.",
    },
    "Bacterial_spot": {
        "urgency": "⚠️ Act within 3-5 days",
        "priority_action": "Spray Copper hydroxide 77WP (3g/L). Switch to drip irrigation.",
        "pesticide_schedule": [
            {"day": "Day 0", "action": "Spray Copper hydroxide 3g/L. Remove overhead irrigation", "type": "Chemical"},
            {"day": "Day 5", "action": "Re-spray copper if rain washed it off. Add Streptomycin 0.5g/L", "type": "Chemical"},
            {"day": "Day 12", "action": "Pseudomonas fluorescens bio-spray 10g/L", "type": "Organic"},
            {"day": "Day 20", "action": "Scout. Preventive copper spray if conditions remain humid", "type": "Preventive"},
        ],
        "organic_alternatives": "Bordeaux mixture 1% + Pseudomonas fluorescens 10g/L spray",
        "dose_reminder": "⚠️ Streptomycin: use sparingly to avoid resistance. Max 2 applications/season.",
        "harvest_interval": "⏰ Copper: 0-day PHI (can harvest same day). Streptomycin: 7-day PHI.",
    },
    "Spider_mites": {
        "urgency": "⚠️ Act within 3-5 days (mites multiply rapidly!)",
        "priority_action": "Spray Abamectin 1.8EC (1ml/L) on leaf undersides. Increase humidity.",
        "pesticide_schedule": [
            {"day": "Day 0", "action": "Abamectin 1ml/L spray focusing on leaf undersides", "type": "Chemical"},
            {"day": "Day 5", "action": "Re-scout. If still present: Spiromesifen 1ml/L (different mode of action)", "type": "Chemical"},
            {"day": "Day 12", "action": "Neem oil 5ml/L organic spray", "type": "Organic"},
            {"day": "Day 20", "action": "Release predatory mites if available (biological control)", "type": "Biological"},
        ],
        "organic_alternatives": "Neem oil 5ml/L + garlic-chili extract spray + predatory mite release",
        "dose_reminder": "⚠️ Rotate miticide classes to prevent resistance. Never use same product >2 consecutive times.",
        "harvest_interval": "⏰ Abamectin: 7-day PHI. Spiromesifen: 3-day PHI.",
    },
    "Yellow_Leaf_Curl_Virus": {
        "urgency": "🚨 URGENT — Remove infected plants NOW!",
        "priority_action": "No cure. Remove & destroy infected plants. Control whiteflies immediately.",
        "pesticide_schedule": [
            {"day": "Day 0", "action": "Remove ALL infected plants. Spray Imidacloprid 0.5ml/L for whitefly control", "type": "Chemical"},
            {"day": "Day 7", "action": "Re-spray whitefly control. Install yellow sticky traps (40/acre)", "type": "Chemical"},
            {"day": "Day 14", "action": "Switch to Thiamethoxam 0.3g/L to prevent resistance", "type": "Chemical"},
            {"day": "Ongoing", "action": "Weekly neem oil spray + sticky trap monitoring", "type": "Organic"},
        ],
        "organic_alternatives": "Yellow sticky traps + silver mulch + neem oil 5ml/L spray weekly",
        "dose_reminder": "⚠️ Virus has NO chemical cure. All treatments target the whitefly vector.",
        "harvest_interval": "⏰ Imidacloprid: 7-day PHI. Thiamethoxam: 5-day PHI.",
    },
    "healthy": {
        "urgency": "✅ No urgent action needed",
        "priority_action": "Maintain preventive care routine.",
        "pesticide_schedule": [
            {"day": "Monthly", "action": "Preventive Bordeaux mixture 1% spray", "type": "Preventive"},
            {"day": "Seasonal", "action": "Soil testing + balanced NPK application", "type": "Nutrition"},
        ],
        "organic_alternatives": "Neem oil monthly spray as general preventive measure",
        "dose_reminder": "✅ Plant is healthy — no chemical treatment needed.",
        "harvest_interval": "✅ No PHI concern — no pesticide applied.",
    },
}

# ============ NEW FEATURE 3: WEATHER-BASED DISEASE FORECAST ============
def get_weather_disease_forecast(weather_data):
    """
    Returns a detailed future disease forecast based on current weather conditions.
    Works with or without real API data.
    """
    if not weather_data:
        # Demo mode — use typical monsoon values
        temp = 28
        humidity = 82
        wind = 3.5
        desc = "Overcast Clouds"
        demo = True
    else:
        temp     = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind     = weather_data['wind']['speed']
        desc     = weather_data['weather'][0]['description'].title()
        demo = False

    forecasts = []

    # ---- Fungal Risk ----
    if humidity >= 85 and temp >= 20:
        forecasts.append({
            "disease": "🍄 Late Blight / Downy Mildew",
            "risk": "🔴 VERY HIGH",
            "risk_level": 3,
            "window": "48-72 hours",
            "reason": f"Humidity {humidity}% + Temp {temp:.1f}°C = Perfect spore germination conditions.",
            "action": "Spray Metalaxyl+Mancozeb TODAY before rain. Do not wait!",
            "crops_at_risk": ["Tomato", "Potato", "Grape", "Cucumber"],
        })
    elif humidity >= 75 and temp >= 18:
        forecasts.append({
            "disease": "🍄 Early Blight / Leaf Spot",
            "risk": "🟠 HIGH",
            "risk_level": 2,
            "window": "3-5 days",
            "reason": f"Humidity {humidity}% + Temp {temp:.1f}°C favors fungal growth.",
            "action": "Preventive Mancozeb spray within 2 days. Monitor daily.",
            "crops_at_risk": ["Tomato", "Potato", "Corn", "Pepper"],
        })
    else:
        forecasts.append({
            "disease": "🍄 Fungal Diseases",
            "risk": "🟢 LOW",
            "risk_level": 0,
            "window": "7+ days",
            "reason": f"Humidity {humidity}% + Temp {temp:.1f}°C — not ideal for most fungi.",
            "action": "Routine monitoring sufficient. No urgent spray needed.",
            "crops_at_risk": [],
        })

    # ---- Bacterial Risk ----
    if humidity >= 80 and temp >= 24 and wind <= 4:
        forecasts.append({
            "disease": "🦠 Bacterial Blight / Spot",
            "risk": "🟠 MEDIUM-HIGH",
            "risk_level": 2,
            "window": "2-4 days",
            "reason": f"High humidity ({humidity}%) + warm ({temp:.1f}°C) + low wind = bacteria-friendly.",
            "action": "Switch to drip irrigation. Preventive copper spray if rain expected.",
            "crops_at_risk": ["Tomato", "Pepper", "Peach", "Bean"],
        })
    else:
        forecasts.append({
            "disease": "🦠 Bacterial Diseases",
            "risk": "🟢 LOW",
            "risk_level": 0,
            "window": "7+ days",
            "reason": "Current conditions less favorable for bacterial spread.",
            "action": "No immediate action required.",
            "crops_at_risk": [],
        })

    # ---- Pest Risk ----
    if temp >= 28 and humidity <= 60:
        forecasts.append({
            "disease": "🕷️ Spider Mites / Thrips",
            "risk": "🔴 HIGH",
            "risk_level": 3,
            "window": "1-3 days",
            "reason": f"High temp ({temp:.1f}°C) + low humidity ({humidity}%) = rapid mite reproduction.",
            "action": "Increase irrigation frequency. Apply Abamectin or neem oil TODAY.",
            "crops_at_risk": ["Tomato", "Bean", "Cucumber", "Cotton"],
        })
    elif temp >= 24:
        forecasts.append({
            "disease": "🦟 Aphids / Whitefly",
            "risk": "🟠 MEDIUM",
            "risk_level": 2,
            "window": "3-5 days",
            "reason": f"Warm temperature ({temp:.1f}°C) accelerates insect life cycles.",
            "action": "Scout undersides of leaves. Install yellow sticky traps.",
            "crops_at_risk": ["Tomato", "Pepper", "Cabbage", "Eggplant"],
        })
    else:
        forecasts.append({
            "disease": "🦟 Insect Pests",
            "risk": "🟢 LOW",
            "risk_level": 0,
            "window": "7+ days",
            "reason": f"Cooler temperature ({temp:.1f}°C) slows insect activity.",
            "action": "Routine scouting only.",
            "crops_at_risk": [],
        })

    # ---- Viral Risk ----
    if temp >= 25 and humidity >= 60:
        forecasts.append({
            "disease": "🧬 Viral Diseases (TYLCV, Mosaic)",
            "risk": "🟠 MEDIUM",
            "risk_level": 2,
            "window": "5-10 days",
            "reason": f"Warm humid conditions favor whitefly/aphid vectors of viruses.",
            "action": "Install sticky traps. Apply Imidacloprid for vector control.",
            "crops_at_risk": ["Tomato", "Pepper", "Squash", "Bean"],
        })

    # ---- Powdery Mildew ----
    if 20 <= temp <= 30 and 40 <= humidity <= 70:
        forecasts.append({
            "disease": "💨 Powdery Mildew",
            "risk": "🟠 MEDIUM",
            "risk_level": 2,
            "window": "3-7 days",
            "reason": f"Moderate humidity ({humidity}%) + temp {temp:.1f}°C = powdery mildew ideal range.",
            "action": "Sulfur spray (3g/L) preventively. Improve air circulation.",
            "crops_at_risk": ["Grapes", "Squash", "Apple", "Rose", "Cucumber"],
        })

    # Sort by risk level (highest first)
    forecasts.sort(key=lambda x: x["risk_level"], reverse=True)

    return forecasts, temp, humidity, wind, desc, demo


# ============ SESSION STATE INIT ============
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'disease_count' not in st.session_state:
    st.session_state.disease_count = 0
if 'healthy_count' not in st.session_state:
    st.session_state.healthy_count = 0
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_detected_disease' not in st.session_state:
    st.session_state.last_detected_disease = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

def predict_frame(frame_bgr, model):
    """Predict disease from a BGR OpenCV frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return predict(image, model)

def get_disease_info(class_name):
    for key, info_tuple in DISEASE_INFO.items():
        if key.lower() in class_name.lower():
            return info_tuple
    return (
        "No additional information available.",
        "Consult an agricultural expert.",
        "Follow general prevention practices.",
        "Consult your local agricultural extension office.",
        "Apply balanced NPK fertilizer as per soil test."
    )

def get_severity(confidence):
    if confidence >= 90:
        return "🔴 High", "red"
    elif confidence >= 70:
        return "🟠 Medium", "orange"
    else:
        return "🟡 Low", "yellow"

# ============ NEW: AI Crop Doctor offline lookup ============
def crop_doctor_answer(query):
    """
    Offline knowledge-base chatbot. Returns doctor KB entry for the best matching disease.
    No internet required.
    """
    query_lower = query.lower().strip()
    best_match = None
    best_score = 0

    for disease_key, info in CROP_DOCTOR_KB.items():
        keywords = disease_key.split()
        score = sum(1 for kw in keywords if kw in query_lower)
        # Also check symptoms text
        if any(word in query_lower for word in disease_key.split()):
            score += 2
        if score > best_score:
            best_score = score
            best_match = (disease_key, info)

    # Check if user asked about pesticide dosage / organic / treatment generically
    if best_match is None or best_score == 0:
        # Try to match from last scanned disease
        if st.session_state.last_detected_disease:
            last = st.session_state.last_detected_disease.lower()
            for disease_key, info in CROP_DOCTOR_KB.items():
                if disease_key in last or any(w in last for w in disease_key.split()):
                    best_match = (disease_key, info)
                    break

    if best_match is None:
        return None, None

    return best_match[0], best_match[1]


# ============ SOIL ANALYSIS ============
def analyze_soil(ph, moisture, nitrogen, phosphorus, potassium, soil_type):
    recommendations = []
    issues = []

    if ph < 5.5:
        issues.append("⚠️ Soil is too acidic (pH < 5.5)")
        recommendations.append("🪨 Apply agricultural lime (CaCO3) at 2-4 tons/acre to raise pH.")
    elif ph > 7.5:
        issues.append("⚠️ Soil is too alkaline (pH > 7.5)")
        recommendations.append("🌋 Apply elemental sulfur at 200-500 kg/acre or use acidifying fertilizers like ammonium sulfate.")
    else:
        recommendations.append("✅ Soil pH is in the optimal range (5.5 - 7.5) for most crops.")

    if moisture < 30:
        issues.append("⚠️ Soil moisture is too low (< 30%)")
        recommendations.append("💧 Increase irrigation frequency. Consider drip irrigation for efficient water use.")
    elif moisture > 80:
        issues.append("⚠️ Soil moisture is too high (> 80%) — risk of root rot and anaerobic conditions.")
        recommendations.append("🚰 Improve field drainage. Reduce irrigation. Add organic matter to improve drainage.")
    else:
        recommendations.append("✅ Soil moisture is adequate.")

    if nitrogen == "Low":
        issues.append("⚠️ Low Nitrogen — expect pale yellow leaves, stunted growth.")
        recommendations.append("🌱 Apply Urea (46-0-0) at 100-150 kg/ha or split dose. Alternatively use ammonium nitrate.")
    elif nitrogen == "High":
        issues.append("⚠️ Excess Nitrogen — may cause lush growth susceptible to disease.")
        recommendations.append("🌿 Reduce nitrogen application. Avoid urea top dressing for now.")
    else:
        recommendations.append("✅ Nitrogen level is optimal.")

    if phosphorus == "Low":
        issues.append("⚠️ Low Phosphorus — poor root development and flowering expected.")
        recommendations.append("🌾 Apply DAP (18-46-0) at 80-100 kg/ha at sowing. Or use Single Super Phosphate.")
    elif phosphorus == "High":
        recommendations.append("ℹ️ High Phosphorus — avoid additional P fertilizer. May lock out Zinc and Iron.")
    else:
        recommendations.append("✅ Phosphorus level is optimal.")

    if potassium == "Low":
        issues.append("⚠️ Low Potassium — plant immunity and fruit quality will suffer.")
        recommendations.append("🍌 Apply Muriate of Potash (MOP) 60-0-60 at 60-80 kg/ha or use SOP for chloride-sensitive crops.")
    elif potassium == "High":
        recommendations.append("ℹ️ High Potassium — may interfere with calcium and magnesium uptake.")
    else:
        recommendations.append("✅ Potassium level is optimal.")

    if soil_type == "Sandy":
        recommendations.append("🏜️ Sandy soil: Add organic compost (5-10 tons/acre). Use slow-release fertilizers. Irrigate more frequently but with less water.")
    elif soil_type == "Clay":
        recommendations.append("🟤 Clay soil: Add gypsum (500kg/acre) to improve structure. Avoid waterlogging. Use raised beds.")
    elif soil_type == "Loamy":
        recommendations.append("✅ Loamy soil: Ideal for most crops. Maintain organic matter with yearly compost applications.")
    elif soil_type == "Black (Regur)":
        recommendations.append("⚫ Black soil: Excellent water retention. Good for cotton, soybean. Avoid waterlogging. Apply zinc micronutrient.")
    elif soil_type == "Red Laterite":
        recommendations.append("🔴 Red laterite: Low fertility. Apply heavy organic manure. Add lime if acidic. Micronutrient supplementation important.")

    return issues, recommendations

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

# ============ VIDEO PROCESSING ============
def process_video_frames(video_path, model, sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    results_list = []
    frame_count = 0
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            try:
                preds = predict_frame(frame, model)
                top_class, top_prob, _ = preds[0]
                parts = top_class.split("___")
                plant = parts[0].replace("_", " ")
                condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                is_healthy = "healthy" in condition.lower()

                color = (0, 255, 0) if is_healthy else (0, 0, 255)
                label = f"{plant}: {condition} ({top_prob:.1f}%)"
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10, 10), (w - 10, 70), color, 2)
                cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frames.append({
                    "frame_num": frame_count,
                    "image": Image.fromarray(frame_rgb),
                    "plant": plant,
                    "condition": condition,
                    "confidence": top_prob,
                    "healthy": is_healthy
                })
                results_list.append((frame_count, plant, condition, top_prob, is_healthy))
            except Exception:
                pass
        frame_count += 1

    cap.release()
    return processed_frames, results_list

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
        margin: 0.5rem 0;
    }
    .treatment-box {
        background: linear-gradient(135deg, #F3E5F5, #E8EAF6);
        border-left: 6px solid #9C27B0;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prevention-box {
        background: linear-gradient(135deg, #E8F5E9, #F9FBE7);
        border-left: 6px solid #8BC34A;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .medicine-box {
        background: linear-gradient(135deg, #FCE4EC, #FFF8E1);
        border-left: 6px solid #E91E63;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .fertilizer-box {
        background: linear-gradient(135deg, #E0F2F1, #E8F5E9);
        border-left: 6px solid #009688;
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
    .soil-issue  { background-color: #FFEBEE; border-left: 5px solid #F44336;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .soil-rec    { background-color: #E8F5E9; border-left: 5px solid #4CAF50;
                   padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0; }
    .prediction-bar {
        background: linear-gradient(90deg, #4CAF50, #81C784);
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px 0;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .camera-box {
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        background: #F1F8E9;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .video-result-card {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    /* New feature styles */
    .doctor-chat-user {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        text-align: right;
        border-right: 4px solid #2196F3;
    }
    .doctor-chat-bot {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-radius: 12px 12px 12px 2px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 4px solid #4CAF50;
    }
    .schedule-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .forecast-high {
        background: linear-gradient(135deg, #FFEBEE, #FCE4EC);
        border-left: 6px solid #F44336;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.6rem 0;
    }
    .forecast-medium {
        background: linear-gradient(135deg, #FFF8E1, #FFF3E0);
        border-left: 6px solid #FF9800;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.6rem 0;
    }
    .forecast-low {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-left: 6px solid #4CAF50;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 0.6rem 0;
    }
    .phi-box {
        background: linear-gradient(135deg, #FFF3E0, #FBE9E7);
        border-left: 5px solid #FF5722;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============ LANGUAGE SELECTOR in SIDEBAR ============
with st.sidebar:
    st.markdown("### 🌾 SmartAgriGuard")
    st.markdown("---")

    st.subheader("🌐 Language / भाषा")
    selected_lang = st.selectbox(
        "Select Language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        label_visibility="collapsed"
    )
    st.session_state.language = selected_lang
    L = LANGUAGES[selected_lang]

    st.markdown("---")

    st.subheader(L["location_label"])
    city = st.text_input(L["city_input"], value="Aurangabad",
                         help="Enter your city for weather & risk analysis")

    st.markdown("---")
    st.subheader(L["session_stats"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_scans}</div><small>{L["scans"]}</small></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#F44336">{st.session_state.disease_count}</div><small>{L["diseased"]}</small></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.healthy_count}</div><small>{L["healthy"]}</small></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(L["supported_plants"])
    plants = ["🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn",
              "🍇 Grape", "🍊 Orange", "🍑 Peach", "🫑 Pepper",
              "🥔 Potato", "🫙 Raspberry", "🫘 Soybean", "🎃 Squash",
              "🍓 Strawberry", "🍅 Tomato"]
    for p in plants:
        st.write(p)

    st.markdown("---")
    st.caption("Model: ResNet18 | Accuracy: ~95% | Classes: 38")

# Get language dict shortcut
L = LANGUAGES[st.session_state.language]

# ============ HEADER ============
st.markdown(f'<p class="main-title">{L["title"]}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{L["subtitle"]}</p>', unsafe_allow_html=True)

# ============ MAIN TABS (original 7 + 3 new) ============
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    L["tab_detection"],
    L["tab_weather"],
    L["tab_soil"],
    L["tab_camera"],
    L["tab_video"],
    L["tab_dashboard"],
    L["tab_history"],
    L["tab_doctor"],
    L["tab_smart_rec"],
    L["tab_weather_pred"],
])

# ===== TAB 1 - DETECTION (unchanged) =====
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader(L["upload_header"])
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        show_gradcam = st.checkbox(L["show_gradcam"], value=True, help=L["gradcam_help"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=L["uploaded_leaf"], use_column_width=True)
            st.success(f'{L["image_loaded"]}: {uploaded_file.name}')

    with col2:
        st.subheader(L["results_header"])

        if uploaded_file is None:
            st.info(L["no_image_msg"])
            st.markdown(L["how_to_use"])
        else:
            if not os.path.exists(MODEL_PATH):
                st.error(L["model_missing"])
            else:
                with st.spinner(L["analyzing"]):
                    model   = load_model()
                    results = predict(image, model)

                top_class, top_prob, top_idx = results[0]
                parts     = top_class.split("___")
                plant     = parts[0].replace("_", " ")
                condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                is_healthy = "healthy" in condition.lower()

                # Save last detected disease for Smart Recommendations tab
                st.session_state.last_detected_disease = top_class

                st.session_state.total_scans += 1
                if is_healthy:
                    st.session_state.healthy_count += 1
                else:
                    st.session_state.disease_count += 1

                if is_healthy:
                    st.markdown(f"""
                    <div class="result-box healthy">
                        <h3>{L["healthy_msg"]}</h3>
                        <p><b>{L["plant_label"]}:</b> {plant}</p>
                        <p><b>{L["confidence_label"]}:</b> {top_prob:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    severity_label, severity_color = get_severity(top_prob)
                    st.markdown(f"""
                    <div class="result-box diseased">
                        <h3>{L["disease_msg"]}</h3>
                        <p><b>{L["plant_label"]}:</b> {plant}</p>
                        <p><b>{L["disease_label"]}:</b> {condition}</p>
                        <p><b>{L["confidence_label"]}:</b> {top_prob:.1f}%</p>
                        <p><b>{L["severity_label"]}:</b> {severity_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                info_tuple = get_disease_info(top_class)
                info       = info_tuple[0]
                treatment  = info_tuple[1]
                prevention = info_tuple[2] if len(info_tuple) > 2 else "Consult agricultural expert."
                medicine   = info_tuple[3] if len(info_tuple) > 3 else "Consult local agri shop."
                fertilizer = info_tuple[4] if len(info_tuple) > 4 else "Apply balanced NPK."

                st.markdown(f'<div class="info-box"><b>{L["about_label"]}:</b><br>{info}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="treatment-box"><b>{L["treatment_label"]}:</b><br>{treatment}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prevention-box"><b>{L["prevention_label"]}:</b><br>{prevention}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="medicine-box"><b>{L["medicine_label"]}:</b><br>{medicine}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fertilizer-box"><b>{L["fertilizer_label"]}:</b><br>{fertilizer}</div>', unsafe_allow_html=True)

                st.write(f"**{L['top3_label']}:**")
                for i, (cls, prob, _) in enumerate(results):
                    p    = cls.split("___")[0].replace("_", " ")
                    c    = cls.split("___")[1].replace("_", " ") if "___" in cls else ""
                    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.progress(int(prob) / 100)
                    st.write(f"{icon} **{p}** — {c} ({prob:.1f}%)")

                if show_gradcam:
                    st.write(f"**{L['gradcam_label']}:**")
                    with st.spinner("Generating heatmap..."):
                        model_grad = load_model()
                        cam_image  = generate_gradcam_image(model_grad, image, top_idx)
                    if cam_image:
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.image(image.resize((224, 224)), caption=L["original_label"], use_column_width=True)
                        with gcol2:
                            st.image(cam_image, caption=L["ai_focus_label"], use_column_width=True)
                        st.caption(L["gradcam_caption"])
                    else:
                        st.info("Grad-CAM not available for this image.")

                st.session_state.history.append({
                    "time"      : datetime.now().strftime("%H:%M:%S"),
                    "date"      : datetime.now().strftime("%d/%m/%Y"),
                    "plant"     : plant,
                    "condition" : condition,
                    "confidence": f"{top_prob:.1f}%",
                    "status"    : "Healthy" if is_healthy else "Diseased"
                })

# ===== TAB 2 - WEATHER (unchanged) =====
with tab2:
    st.subheader(L["weather_header"])
    weather_data = get_weather(city)

    if weather_data is None and WEATHER_API_KEY == "YOUR_API_KEY_HERE":
        st.warning("⚠️ Weather API key not set! Get a free key from https://openweathermap.org/api and replace YOUR_API_KEY_HERE in the code.")
        st.info("👉 https://openweathermap.org/api — Free API key available. Then update WEATHER_API_KEY in the code.")

        st.markdown(f"### 📖 {L['demo_mode']}")
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
        temp     = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        desc     = weather_data['weather'][0]['description'].title()
        wind     = weather_data['wind']['speed']
        feels    = weather_data['main']['feels_like']

        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            st.markdown(f'<div class="weather-card">🌡️<br><b>{temp:.1f}°C</b><br><small>Temperature</small></div>', unsafe_allow_html=True)
        with w2:
            st.markdown(f'<div class="weather-card">💧<br><b>{humidity}%</b><br><small>Humidity</small></div>', unsafe_allow_html=True)
        with w3:
            st.markdown(f'<div class="weather-card">🌤️<br><b>{desc}</b><br><small>Condition</small></div>', unsafe_allow_html=True)
        with w4:
            st.markdown(f'<div class="weather-card">💨<br><b>{wind} m/s</b><br><small>Wind Speed</small></div>', unsafe_allow_html=True)
        with w5:
            st.markdown(f'<div class="weather-card">🤔<br><b>{feels:.1f}°C</b><br><small>Feels Like</small></div>', unsafe_allow_html=True)

        st.markdown(f"### {L['risk_header']}")
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

        st.markdown(f"### {L['farming_tips']}")
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

# ===== TAB 3 - SOIL ANALYSIS (unchanged) =====
with tab3:
    st.subheader(L["soil_header"])
    st.info("📋 Enter your soil test values manually. Based on your inputs, we'll generate specific fertilizer and amendment recommendations.")

    with st.form("soil_form"):
        scol1, scol2 = st.columns(2)
        with scol1:
            ph       = st.slider(L["soil_ph"], min_value=3.0, max_value=10.0, value=6.5, step=0.1)
            moisture = st.slider(L["soil_moisture"], min_value=0, max_value=100, value=50, step=1)
            nitrogen = st.selectbox(L["soil_nitrogen"], ["Low", "Optimal", "High"])
        with scol2:
            phosphorus = st.selectbox(L["soil_phosphorus"], ["Low", "Optimal", "High"])
            potassium  = st.selectbox(L["soil_potassium"], ["Low", "Optimal", "High"])
            soil_type  = st.selectbox(L["soil_type"], ["Loamy", "Sandy", "Clay", "Black (Regur)", "Red Laterite"])

        crop_type = st.selectbox("🌾 Primary Crop", [
            "Tomato", "Potato", "Wheat", "Rice", "Corn (Maize)", "Soybean",
            "Cotton", "Sugarcane", "Grapes", "Apple", "Onion", "Other"
        ])
        submitted = st.form_submit_button(L["analyze_soil_btn"], use_container_width=True)

    if submitted:
        st.markdown("---")
        st.markdown("### 🧪 Soil Analysis Report")

        ph_score    = 100 - abs(ph - 6.5) * 20
        moist_score = 100 - abs(moisture - 55) * 2
        n_score     = 100 if nitrogen == "Optimal" else 50
        p_score     = 100 if phosphorus == "Optimal" else 50
        k_score     = 100 if potassium == "Optimal" else 50
        overall     = int((ph_score + moist_score + n_score + p_score + k_score) / 5)
        overall     = max(0, min(100, overall))

        m1, m2, m3 = st.columns(3)
        with m1:
            color = "#4CAF50" if overall >= 70 else "#FF9800" if overall >= 40 else "#F44336"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:{color}">{overall}%</div>
                <div>Overall Soil Health Score</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            ph_status = "Optimal ✅" if 5.5 <= ph <= 7.5 else ("Too Acidic ⚠️" if ph < 5.5 else "Too Alkaline ⚠️")
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:1.4rem">{ph}</div><div>pH — {ph_status}</div></div>', unsafe_allow_html=True)
        with m3:
            moist_status = "Good ✅" if 30 <= moisture <= 80 else ("Too Dry ⚠️" if moisture < 30 else "Too Wet ⚠️")
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:1.4rem">{moisture}%</div><div>Moisture — {moist_status}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        issues, recommendations = analyze_soil(ph, moisture, nitrogen, phosphorus, potassium, soil_type)

        if issues:
            st.markdown("#### ⚠️ Issues Found")
            for issue in issues:
                st.markdown(f'<div class="soil-issue">{issue}</div>', unsafe_allow_html=True)

        st.markdown("#### ✅ Recommendations")
        for rec in recommendations:
            st.markdown(f'<div class="soil-rec">{rec}</div>', unsafe_allow_html=True)

        st.markdown(f"#### 🌾 Specific Advice for {crop_type}")
        crop_advice = {
            "Tomato":    "Tomatoes need high Potassium for fruit quality. Apply Calcium Nitrate foliar spray to prevent Blossom End Rot. Target pH 6.0-6.8.",
            "Potato":    "Potatoes prefer slightly acidic soil (pH 5.0-6.0). High potassium (MOP) improves tuber size. Avoid fresh manure — use composted manure.",
            "Wheat":     "Wheat performs best at pH 6.0-7.0. Apply Zinc (ZnSO4 25kg/ha) and adequate phosphorus at sowing. Split urea application recommended.",
            "Rice":      "Rice prefers slightly acidic soil (pH 5.5-6.5). Flood irrigation needed. Apply zinc sulfate 25kg/ha for Khaira disease prevention.",
            "Corn (Maize)": "Corn requires high nitrogen. Apply 120-150 kg N/ha in splits. Zinc deficiency common — apply ZnSO4. Target pH 6.0-7.0.",
            "Soybean":   "Soybean fixes nitrogen — no extra N needed. High phosphorus requirement. Apply Rhizobium inoculant to seeds. Target pH 6.0-7.0.",
            "Cotton":    "Cotton prefers neutral pH (6.0-7.5). High nitrogen + potassium for boll development. Black soil (Regur) is ideal.",
            "Sugarcane": "Sugarcane needs high nitrogen and potassium. Apply in 3 splits. Trashching mulch conserves moisture. Target pH 6.0-7.5.",
            "Grapes":    "Grapes prefer slightly acidic soil (pH 5.5-6.5). High potassium for fruit sweetness. Magnesium deficiency common on sandy soils.",
            "Apple":     "Apple requires pH 6.0-7.0. Boron deficiency causes corky fruit — apply Borax 1kg/100L water. High organic matter preferred.",
            "Onion":     "Onion prefers pH 6.0-7.0. Sulfur fertilizer improves flavor and pungency. High phosphorus for root development.",
            "Other":     "Follow general good agricultural practices. Get a detailed soil test from your nearest Krishi Vigyan Kendra (KVK) for crop-specific recommendations."
        }
        st.info(crop_advice.get(crop_type, crop_advice["Other"]))

# ===== TAB 4 - LIVE CAMERA (unchanged) =====
with tab4:
    st.subheader(L["camera_header"])

    st.info("""
    📷 **How Live Camera Detection Works:**
    - Click **'Take Photo'** to capture a leaf image using your device camera
    - The AI will instantly detect disease in the captured image
    - Results appear with full disease info, treatment, and Grad-CAM heatmap
    - Works on mobile and desktop browsers
    """)

    camera_image = st.camera_input("📸 Point camera at a leaf and take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="📸 Captured Leaf Image", width=300)

        if not os.path.exists(MODEL_PATH):
            st.error(L["model_missing"])
        else:
            with st.spinner(L["analyzing"]):
                model   = load_model()
                results = predict(image, model)

            top_class, top_prob, top_idx = results[0]
            parts     = top_class.split("___")
            plant     = parts[0].replace("_", " ")
            condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            is_healthy = "healthy" in condition.lower()

            st.session_state.last_detected_disease = top_class
            st.session_state.total_scans += 1
            if is_healthy:
                st.session_state.healthy_count += 1
            else:
                st.session_state.disease_count += 1

            if is_healthy:
                st.success(f"✅ **{plant}** — {condition} ({top_prob:.1f}% confidence)")
            else:
                severity_label, _ = get_severity(top_prob)
                st.error(f"⚠️ **{plant}** — {condition} ({top_prob:.1f}%) | Severity: {severity_label}")

            info_tuple = get_disease_info(top_class)
            st.markdown(f'<div class="info-box"><b>{L["about_label"]}:</b><br>{info_tuple[0]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="treatment-box"><b>{L["treatment_label"]}:</b><br>{info_tuple[1]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prevention-box"><b>{L["prevention_label"]}:</b><br>{info_tuple[2]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="medicine-box"><b>{L["medicine_label"]}:</b><br>{info_tuple[3]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="fertilizer-box"><b>{L["fertilizer_label"]}:</b><br>{info_tuple[4]}</div>', unsafe_allow_html=True)

            st.write(f"**{L['top3_label']}:**")
            for i, (cls, prob, _) in enumerate(results):
                p    = cls.split("___")[0].replace("_", " ")
                c    = cls.split("___")[1].replace("_", " ") if "___" in cls else ""
                icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.progress(int(prob) / 100)
                st.write(f"{icon} **{p}** — {c} ({prob:.1f}%)")

            st.write(f"**{L['gradcam_label']}:**")
            with st.spinner("Generating heatmap..."):
                model_grad = load_model()
                cam_image  = generate_gradcam_image(model_grad, image, top_idx)
            if cam_image:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(image.resize((224, 224)), caption=L["original_label"], use_column_width=True)
                with gcol2:
                    st.image(cam_image, caption=L["ai_focus_label"], use_column_width=True)
                st.caption(L["gradcam_caption"])

            st.session_state.history.append({
                "time"      : datetime.now().strftime("%H:%M:%S"),
                "date"      : datetime.now().strftime("%d/%m/%Y"),
                "plant"     : plant,
                "condition" : condition,
                "confidence": f"{top_prob:.1f}%",
                "status"    : "Healthy" if is_healthy else "Diseased",
                "source"    : "Camera"
            })

# ===== TAB 5 - VIDEO DETECTION (unchanged) =====
with tab5:
    st.subheader(L["video_header"])

    st.info("""
    🎬 **Video Frame Detection:**
    - Upload a video of your crop/plants
    - AI analyzes frames automatically (every 30th frame)
    - Shows disease detection results with bounding box overlays
    - Displays frame-by-frame results with disease summary
    """)

    uploaded_video = st.file_uploader(L["upload_video"], type=["mp4", "avi", "mov", "mkv"])
    sample_rate    = st.slider("⚙️ Analyze every N-th frame (lower = more thorough, slower)", 10, 60, 30)

    if uploaded_video is not None:
        if not os.path.exists(MODEL_PATH):
            st.error(L["model_missing"])
        else:
            if st.button(L["process_video"], use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    tmp_path = tmp.name

                with st.spinner("🎬 Processing video frames... This may take a moment."):
                    model = load_model()
                    processed_frames, results_list = process_video_frames(tmp_path, model, sample_rate)

                os.unlink(tmp_path)

                if not processed_frames:
                    st.warning("⚠️ No frames could be processed. Check if the video file is valid.")
                else:
                    st.success(f"✅ Processed {len(processed_frames)} frames from video!")

                    total_frames    = len(results_list)
                    healthy_frames  = sum(1 for r in results_list if r[4])
                    diseased_frames = total_frames - healthy_frames

                    vc1, vc2, vc3 = st.columns(3)
                    with vc1:
                        st.metric("📹 Frames Analyzed", total_frames)
                    with vc2:
                        st.metric("✅ Healthy Frames", healthy_frames)
                    with vc3:
                        st.metric("⚠️ Diseased Frames", diseased_frames)

                    st.markdown("### 🎞️ Frame-by-Frame Results")

                    disease_counts = {}
                    for _, plant, cond, conf, healthy in results_list:
                        key = f"{plant} — {cond}"
                        if key not in disease_counts:
                            disease_counts[key] = {"count": 0, "avg_conf": 0}
                        disease_counts[key]["count"] += 1
                        disease_counts[key]["avg_conf"] += conf

                    st.markdown("#### 📊 Disease Summary")
                    for key, val in sorted(disease_counts.items(), key=lambda x: -x[1]["count"]):
                        avg_conf = val["avg_conf"] / val["count"]
                        icon = "✅" if "healthy" in key.lower() else "⚠️"
                        st.markdown(f"""
                        <div class="video-result-card">
                            {icon} <b>{key}</b> — {val['count']} frames | Avg confidence: {avg_conf:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 🖼️ Analyzed Frames (with AI overlay)")
                    show_count = min(6, len(processed_frames))
                    cols = st.columns(3)
                    for i, frame_data in enumerate(processed_frames[:show_count]):
                        with cols[i % 3]:
                            status_icon = "✅" if frame_data["healthy"] else "⚠️"
                            caption = f"{status_icon} Frame {frame_data['frame_num']}: {frame_data['plant']} — {frame_data['condition']} ({frame_data['confidence']:.1f}%)"
                            st.image(frame_data["image"], caption=caption, use_column_width=True)

# ===== TAB 6 - DASHBOARD (unchanged) =====
with tab6:
    st.subheader(L["dashboard_header"])

    if st.session_state.total_scans == 0:
        st.info(L["no_scans"])
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_scans}</div><div>{L["total_scans"]}</div></div>', unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#F44336">{st.session_state.disease_count}</div><div>{L["diseases_found"]}</div></div>', unsafe_allow_html=True)
        with d3:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.healthy_count}</div><div>{L["healthy_plants"]}</div></div>', unsafe_allow_html=True)
        with d4:
            rate = (st.session_state.healthy_count / st.session_state.total_scans * 100 if st.session_state.total_scans > 0 else 0)
            st.markdown(f'<div class="stat-card"><div class="stat-number">{rate:.0f}%</div><div>{L["health_rate"]}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        if len(st.session_state.history) > 0:
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("**🥧 Disease vs Healthy Ratio**")
                fig, ax = plt.subplots(figsize=(5, 4))
                labels  = ['Diseased 🦠', 'Healthy ✅']
                sizes   = [st.session_state.disease_count, st.session_state.healthy_count]
                colors  = ['#FF6B6B', '#4CAF50']
                explode = (0.05, 0.05)
                if sum(sizes) > 0:
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                           autopct='%1.1f%%', shadow=True, startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.info("No data yet.")
                plt.close()

            with chart_col2:
                st.write("**📋 Recent Scan History**")
                if st.session_state.history:
                    recent = st.session_state.history[-10:][::-1]
                    for entry in recent:
                        icon = "✅" if entry["status"] == "Healthy" else "⚠️"
                        st.markdown(f"""
                        <div class="video-result-card">
                            {icon} <b>{entry['plant']}</b> — {entry['condition']}
                            <small style="color:#888"> | {entry['confidence']} | {entry['date']} {entry['time']}</small>
                        </div>
                        """, unsafe_allow_html=True)

# ===== TAB 7 - SCAN HISTORY (unchanged) =====
with tab7:
    st.subheader(L["history_header"])

    if not st.session_state.history:
        st.info(L["no_history"])
    else:
        if st.button(L["clear_history"]):
            st.session_state.history = []
            st.rerun()

        st.write(f"**Total records: {len(st.session_state.history)}**")

        for i, entry in enumerate(reversed(st.session_state.history)):
            icon = "✅" if entry["status"] == "Healthy" else "⚠️"
            source = entry.get("source", "Upload")
            st.markdown(f"""
            <div class="video-result-card">
                {icon} <b>#{len(st.session_state.history)-i}</b> &nbsp;
                🌱 <b>{entry['plant']}</b> — {entry['condition']} &nbsp;|&nbsp;
                📊 {entry['confidence']} &nbsp;|&nbsp;
                📅 {entry['date']} {entry['time']} &nbsp;|&nbsp;
                📷 {source}
            </div>
            """, unsafe_allow_html=True)

# ===========================
# ===== TAB 8 — NEW: AI CROP DOCTOR (Offline Chatbot) =====
# ===========================
with tab8:
    st.subheader("🩺 AI Crop Doctor — Expert Knowledge Chatbot")
    st.markdown("""
    <div class="info-box">
        <b>💡 Works completely offline — no internet needed!</b><br>
        Ask me about any plant disease: symptoms, pesticide names, dosage, organic solutions, nearby shop tips.<br>
        <small>Examples: "how to treat late blight", "organic remedy for powdery mildew", "dosage for spider mites", "what is bacterial spot"</small>
    </div>
    """, unsafe_allow_html=True)

    # Show last detected disease as context
    if st.session_state.last_detected_disease:
        parts_d = st.session_state.last_detected_disease.split("___")
        plant_d = parts_d[0].replace("_", " ")
        cond_d  = parts_d[1].replace("_", " ") if len(parts_d) > 1 else ""
        st.info(f"🔬 Last scanned: **{plant_d} — {cond_d}**. You can ask about this disease directly!")

    # Quick question buttons
    st.write("**⚡ Quick Questions:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    quick_q = None
    with qcol1:
        if st.button("💊 Pesticide dosage", use_container_width=True):
            quick_q = "pesticide dosage for last detected disease"
    with qcol2:
        if st.button("🌿 Organic remedy", use_container_width=True):
            quick_q = "organic remedy for last detected disease"
    with qcol3:
        if st.button("⚠️ Severity & urgency", use_container_width=True):
            quick_q = "severity and urgency for last detected disease"
    with qcol4:
        if st.button("🏪 Where to buy", use_container_width=True):
            quick_q = "where to buy medicine for last detected disease"

    # Chat input
    user_input = st.chat_input("Ask the AI Crop Doctor... (e.g. 'how to treat late blight on tomato')")

    # Handle quick questions
    if quick_q:
        # Map to last disease
        if st.session_state.last_detected_disease:
            last = st.session_state.last_detected_disease.lower()
            user_input = quick_q + " " + last
        else:
            user_input = quick_q

    # Process input
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        disease_key, kb_info = crop_doctor_answer(user_input)

        if kb_info is None:
            bot_response = (
                "🤔 I couldn't find specific info for that query in my knowledge base.\n\n"
                "**Try asking about specific diseases like:**\n"
                "- late blight, early blight, powdery mildew\n"
                "- bacterial spot, spider mites, leaf mold\n"
                "- yellow leaf curl virus, mosaic virus\n\n"
                "Or **scan a leaf first** using the Disease Detection tab, then ask about it here!"
            )
            st.session_state.chat_history.append({"role": "bot", "content": bot_response, "kb": None, "disease": None})
        else:
            bot_response = f"Found information for: **{disease_key.title()}**"
            st.session_state.chat_history.append({"role": "bot", "content": bot_response, "kb": kb_info, "disease": disease_key})

    # Display chat history
    st.markdown("---")
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; color:#999; padding:2rem;">
            🩺 <b>Ask your first question above!</b><br>
            <small>The AI Crop Doctor answers 100% offline using expert agricultural knowledge.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="doctor-chat-user">👨‍🌾 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                if msg.get("kb") is None:
                    st.markdown(f'<div class="doctor-chat-bot">🩺 <b>AI Doctor:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    kb   = msg["kb"]
                    dkey = msg["disease"]

                    with st.container():
                        st.markdown(f'<div class="doctor-chat-bot">🩺 <b>AI Doctor — {dkey.replace("_", " ").title()}</b>', unsafe_allow_html=True)

                        # Severity
                        sev = kb.get("severity", "")
                        sev_color = "forecast-high" if "HIGH" in sev or "CRITICAL" in sev else ("forecast-medium" if "MEDIUM" in sev else "forecast-low")
                        st.markdown(f'<div class="{sev_color}"><b>⚡ Severity:</b> {sev}</div>', unsafe_allow_html=True)

                        # Symptoms
                        st.markdown(f'<div class="info-box"><b>🔍 Symptoms:</b><br>{kb.get("symptoms", "N/A")}</div>', unsafe_allow_html=True)

                        # Chemical treatments
                        if kb.get("chemical"):
                            st.markdown("**💉 Chemical Treatments (Pesticide + Dosage):**")
                            for chem in kb["chemical"]:
                                st.markdown(f"""
                                <div class="medicine-box">
                                    <b>{chem['name']}</b><br>
                                    📏 Dose: <b>{chem['dose']}</b> &nbsp;|&nbsp; 🔁 Interval: <b>{chem['interval']}</b>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("✅ No chemical treatment needed.")

                        # Organic alternatives
                        st.markdown("**🌿 Organic / Jaivik (जैविक) Remedies:**")
                        st.markdown('<div class="prevention-box">' + "<br>".join(kb.get("organic", [])) + "</div>", unsafe_allow_html=True)

                        # Shop hint
                        st.markdown(f'<div class="fertilizer-box"><b>🏪 Where to Buy:</b><br>{kb.get("shop_hint", "Contact local agri shop or KVK.")}</div>', unsafe_allow_html=True)

                        st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ===========================
# ===== TAB 9 — NEW: SMART RECOMMENDATION SYSTEM =====
# ===========================
with tab9:
    st.subheader("💡 Smart Recommendation System — What To Do Next")
    st.markdown("""
    <div class="info-box">
        <b>🎯 AI-powered action planner based on your last disease scan.</b><br>
        Get a step-by-step pesticide schedule, exact dosage, organic alternatives, and Pre-Harvest Interval (PHI) reminders.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_detected_disease is None:
        st.warning("⚠️ No disease scanned yet! Go to the **Disease Detection** tab, scan a leaf, then come back here.")
        st.info("💡 After scanning, this tab will show you exactly: what pesticide to use, how much, when to spray, organic options, and when it's safe to harvest.")
    else:
        parts_r  = st.session_state.last_detected_disease.split("___")
        plant_r  = parts_r[0].replace("_", " ")
        cond_r   = parts_r[1].replace("_", " ") if len(parts_r) > 1 else "Unknown"
        is_h_r   = "healthy" in cond_r.lower()

        st.markdown(f"### 🌱 Recommendations for: **{plant_r} — {cond_r}**")
        st.markdown("---")

        # Find best matching recommendation
        rec_data = None
        for key in SMART_RECOMMENDATIONS:
            if key.lower() in st.session_state.last_detected_disease.lower() or \
               any(w in st.session_state.last_detected_disease.lower() for w in key.lower().split("_")):
                rec_data = SMART_RECOMMENDATIONS[key]
                break

        if rec_data is None:
            rec_data = SMART_RECOMMENDATIONS.get("healthy", None)

        if rec_data:
            # Urgency banner
            urgency  = rec_data["urgency"]
            urg_class = "forecast-high" if "URGENT" in urgency else ("forecast-medium" if "Act within" in urgency else "forecast-low")
            st.markdown(f'<div class="{urg_class}"><h4>{urgency}</h4><p>{rec_data["priority_action"]}</p></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Spray Schedule
            st.markdown("### 📅 Step-by-Step Spray Schedule")
            for step in rec_data["pesticide_schedule"]:
                type_color = {
                    "Chemical": "#FCE4EC",
                    "Organic": "#E8F5E9",
                    "Biological": "#E3F2FD",
                    "Preventive": "#FFF8E1",
                    "Nutrition": "#F3E5F5",
                }.get(step["type"], "#F5F5F5")
                type_icon = {
                    "Chemical": "💉",
                    "Organic": "🌿",
                    "Biological": "🦠",
                    "Preventive": "🛡️",
                    "Nutrition": "🌱",
                }.get(step["type"], "📋")
                st.markdown(f"""
                <div class="schedule-card" style="background:{type_color};">
                    <b>📆 {step['day']}</b> &nbsp; {type_icon} <b>{step['type']}</b><br>
                    {step['action']}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Two column layout for organic + PHI
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown("### 🌿 Organic Alternative")
                st.markdown(f'<div class="prevention-box">{rec_data["organic_alternatives"]}</div>', unsafe_allow_html=True)

                st.markdown("### ⚠️ Dosage Reminder")
                st.markdown(f'<div class="medicine-box">{rec_data["dose_reminder"]}</div>', unsafe_allow_html=True)

            with rec_col2:
                st.markdown("### ⏰ Pre-Harvest Interval (PHI)")
                st.markdown(f'<div class="phi-box"><b>🚫 Do NOT harvest before PHI expires!</b><br><br>{rec_data["harvest_interval"]}</div>', unsafe_allow_html=True)

                st.markdown("### 📞 Get Expert Help")
                st.markdown("""
                <div class="info-box">
                    <b>🏛️ Free Government Resources:</b><br>
                    📞 Kisan Call Centre: <b>1800-180-1551</b> (Toll Free)<br>
                    🏫 Nearest KVK (Krishi Vigyan Kendra)<br>
                    🌐 <a href="https://farmer.gov.in" target="_blank">farmer.gov.in</a><br>
                    📱 mKisan SMS Portal for free agri advice
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific recommendation found. Please consult your local agricultural extension officer.")

# ===========================
# ===== TAB 10 — NEW: WEATHER-BASED DISEASE FORECAST =====
# ===========================
with tab10:
    st.subheader("🔮 Weather-Based Disease Forecast")
    st.markdown("""
    <div class="info-box">
        <b>📡 AI predicts which diseases are likely to appear in the next 1-7 days based on current weather.</b><br>
        Get ahead of disease outbreaks before they happen!
    </div>
    """, unsafe_allow_html=True)

    weather_data_f = get_weather(city)
    forecasts, temp_f, humidity_f, wind_f, desc_f, demo_f = get_weather_disease_forecast(weather_data_f)

    # Weather summary bar
    if demo_f:
        st.warning("⚠️ No Weather API key set — showing forecast based on typical monsoon conditions (Demo Mode). Add your OpenWeatherMap API key for real data.")
    
    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        st.markdown(f'<div class="weather-card">🌡️<br><b>{temp_f:.1f}°C</b><br><small>Temperature</small></div>', unsafe_allow_html=True)
    with wf2:
        st.markdown(f'<div class="weather-card">💧<br><b>{humidity_f}%</b><br><small>Humidity</small></div>', unsafe_allow_html=True)
    with wf3:
        st.markdown(f'<div class="weather-card">💨<br><b>{wind_f} m/s</b><br><small>Wind</small></div>', unsafe_allow_html=True)
    with wf4:
        st.markdown(f'<div class="weather-card">🌤️<br><b>{desc_f}</b><br><small>Condition</small></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Disease Risk Forecast — Next 7 Days")

    for fc in forecasts:
        risk_text = fc["risk"]
        if "VERY HIGH" in risk_text or "HIGH" in risk_text:
            css = "forecast-high"
        elif "MEDIUM" in risk_text or "MEDIUM-HIGH" in risk_text:
            css = "forecast-medium"
        else:
            css = "forecast-low"

        crops_str = ""
        if fc["crops_at_risk"]:
            crops_str = f"<br>🌾 <b>Crops at risk:</b> {', '.join(fc['crops_at_risk'])}"

        st.markdown(f"""
        <div class="{css}">
            <h4>{fc['disease']} &nbsp; {risk_text}</h4>
            <p>⏱️ <b>Expected window:</b> {fc['window']}</p>
            <p>🔬 <b>Why:</b> {fc['reason']}</p>
            <p>✅ <b>Action:</b> {fc['action']}</p>
            {crops_str}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Today's Preventive Action Checklist")

    # Dynamic checklist based on risk levels
    high_risks = [fc for fc in forecasts if fc["risk_level"] >= 3]
    med_risks  = [fc for fc in forecasts if fc["risk_level"] == 2]

    if high_risks:
        st.error("🚨 **HIGH RISK CONDITIONS DETECTED — Immediate Action Needed:**")
        for fc in high_risks:
            st.markdown(f"- {fc['action']}")

    if med_risks:
        st.warning("⚠️ **MEDIUM RISK — Preventive measures recommended:**")
        for fc in med_risks:
            st.markdown(f"- {fc['action']}")

    if not high_risks and not med_risks:
        st.success("✅ **Low disease risk today!** Continue normal monitoring schedule.")

    # General tips based on conditions
    st.markdown("### 🌾 General Farm Tips Based on Today's Weather")
    tips = []
    if humidity_f > 80:
        tips.append("💧 **High humidity** — Avoid overhead irrigation. Spray fungicide preventively.")
    if temp_f > 30:
        tips.append("☀️ **High temperature** — Water crops early morning or evening only.")
    if temp_f > 28 and humidity_f < 60:
        tips.append("🕷️ **Mite weather** — Check leaf undersides. Keep soil moist to raise humidity.")
    if wind_f > 6:
        tips.append("💨 **High wind** — Postpone spray operations. Disease spores may spread rapidly.")
    if humidity_f >= 75 and temp_f >= 18:
        tips.append("🍄 **Fungal alert** — Apply preventive fungicide spray if not done in last 7 days.")
    if not tips:
        tips.append("✅ Conditions are favorable. Maintain regular monitoring and irrigation schedule.")

    for tip in tips:
        st.markdown(f"- {tip}")