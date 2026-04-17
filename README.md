# 🌾 SmartAgriGuard
### AI-Powered Crop Disease Detection & Farm Intelligence System

> Detect plant diseases instantly, get treatment plans, forecast disease outbreaks, and manage your farm smarter — all powered by deep learning.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~95%+ |
| Model Architecture | ResNet-18 (timm) |
| Total Classes | 38 plant disease categories |
| Dataset | PlantVillage (54,305 images) |
| Device Support | CPU & CUDA GPU |

---

## 🌱 Supported Plants & Diseases

| Plant | Diseases Detected |
|---|---|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍊 Orange | Haunglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper (Bell) | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫙 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smartagriguard.git
cd smartagriguard
```

### 2. Create Virtual Environment
```bash
python -m venv hvenv
# Windows
hvenv\Scripts\activate
# Linux / macOS
source hvenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Trained Model
Place `plant_disease_model.pth` in the project root folder:
```
plant disease detection/
└── plant_disease_model.pth   ← place here
```

### 5. (Optional) Add Weather API Key
Edit `app.py` and replace:
```python
WEATHER_API_KEY = "YOUR_API_KEY_HERE"
```
with your free key from [openweathermap.org](https://openweathermap.org/api).

> Without a key, the app runs in **Demo Mode** — all features work except live weather.

### 6. Run the Web App
```bash
streamlit run app.py
```
Open your browser at: **http://localhost:8503/**

### 7. Terminal Prediction (Optional)
```bash
python predict.py --image path/to/leaf.jpg
```

---

## ✨ Features

### 🔬 Tab 1 — Disease Detection
- Upload any leaf image (JPG/PNG)
- Instant AI classification across 38 classes
- Confidence score + severity rating (High / Medium / Low)
- Top-3 predictions with progress bars
- Full disease info: description, treatment, prevention, medicine, fertilizer advice
- **Grad-CAM heatmap** — visualizes where AI is "looking" on the leaf

### 🌦️ Tab 2 — Weather & Disease Risk
- Live weather data for your city (requires OpenWeatherMap API key)
- Automatic risk assessment: Fungal, Bacterial, Pest activity
- Today's farming recommendations based on humidity, temperature, and wind

### 🌱 Tab 3 — Soil Analysis
- Enter soil test values: pH, moisture, N/P/K levels, soil type
- Overall Soil Health Score (0–100%)
- Specific fertilizer & amendment recommendations per nutrient
- Crop-specific advice for 12 major crops (Tomato, Wheat, Rice, Cotton, etc.)

### 📷 Tab 4 — Live Camera Detection
- Use your device camera to take a photo of a leaf
- Instant disease detection on captured image
- Full results including Grad-CAM heatmap

### 🎬 Tab 5 — Video Detection
- Upload a crop video (MP4, AVI, MOV, MKV)
- AI analyzes every N-th frame (configurable)
- Frame-by-frame results with AI bounding box overlay
- Summary: healthy vs diseased frame counts

### 📊 Tab 6 — Farm Analytics Dashboard
- Session statistics: total scans, diseases found, health rate
- Pie chart: Disease vs Healthy ratio
- Recent scan history in reverse chronological order

### 📋 Tab 7 — Scan History
- Complete log of all scans with plant, condition, confidence, date/time, and source
- Clear history button

### 🩺 Tab 8 — AI Crop Doctor (Offline Chatbot)
- **100% offline** — no internet required
- Chat interface to ask about any disease
- Provides: symptoms, pesticide names + exact dosage, organic/jaivik remedies, where to buy medicines
- Quick-action buttons: Pesticide dosage / Organic remedy / Severity / Where to buy
- Automatically uses last scanned disease as context

### 💡 Tab 9 — Smart Recommendation System
- Personalized action plan based on your last disease scan
- Step-by-step pesticide spray schedule (Day 0, Day 7, Day 14, Day 21)
- Organic alternative plans
- Dosage reminders and **Pre-Harvest Interval (PHI)** warnings
- Free government helpline references (Kisan Call Centre: 1800-180-1551)

### 🔮 Tab 10 — Weather-Based Disease Forecast
- Predicts which diseases are likely to appear in the next 1–7 days
- Based on real-time temperature, humidity, and wind conditions
- Covers: Fungal, Bacterial, Pest, Viral, and Powdery Mildew risk
- Color-coded risk levels: 🔴 Very High / 🟠 High / 🟡 Medium / 🟢 Low
- Actionable preventive checklist

---

## 🌐 Multi-Language Support

| Language | Script |
|---|---|
| English | Latin |
| मराठी (Marathi) | Devanagari |
| हिंदी (Hindi) | Devanagari |

Switch language from the sidebar dropdown — all UI labels, buttons, and messages translate instantly.

---

## 📁 Project Structure

```
plant disease detection/
├── app.py                      ← Main Streamlit web application
├── predict.py                  ← Command-line leaf prediction script
├── train.py                    ← Model training script
├── check_accuracy.py           ← Accuracy evaluation script
├── plant_disease_model.pth     ← Trained ResNet-18 model weights
├── requirements.txt            ← Python dependencies
├── .env                        ← Environment variables (API keys)
├── .gitignore
├── data/                       ← Dataset folder
└── utils/                      ← Utility modules
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Model Architecture | ResNet-18 (via `timm`) |
| Web Application | Streamlit |
| Computer Vision | OpenCV, PIL |
| Explainability | Grad-CAM |
| Visualization | Matplotlib |
| Weather API | OpenWeatherMap |
| Language | Python 3.8+ |

---

## 🧠 Model Training (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload `train.py` or paste training code
4. Train on the PlantVillage dataset
5. Download `plant_disease_model.pth` from Google Drive and place it in the project root

---

## 🔑 Environment Variables

Create a `.env` file in the project root (already listed in `.gitignore`):

```
WEATHER_API_KEY=your_openweathermap_api_key_here
```

Or edit `app.py` directly:
```python
WEATHER_API_KEY = "your_key_here"
```

Get a free API key at: https://openweathermap.org/api

---

## 📸 How to Use — Step by Step

1. Run `streamlit run app.py` in your terminal
2. Open `http://localhost:8501` in your browser
3. Select your language from the sidebar (English / मराठी / हिंदी)
4. Enter your city name for weather-based disease risk
5. Go to **Disease Detection** tab → Upload a leaf image
6. View diagnosis: plant name, disease, confidence, treatment plan, and Grad-CAM heatmap
7. Visit **AI Crop Doctor** tab to ask follow-up questions about dosage or organic options
8. Check **Smart Recommendations** for a day-wise spray schedule
9. Visit **Disease Forecast** tab to see upcoming disease risks based on weather

---

## ⚠️ Important Notes

- `plant_disease_model.pth` (~43 MB) is **not included** in the repository. You must add it manually.
- The model is trained on the **PlantVillage dataset** — it performs best on clean, well-lit leaf images against a plain background.
- The AI Crop Doctor (Tab 8) works **completely offline** — no internet needed.
- Weather-related features (Tabs 2 and 10) require a free OpenWeatherMap API key for live data; they fall back to demo mode without one.
- Pre-Harvest Interval (PHI) reminders are provided as a guide — always verify with your local agricultural extension officer before spraying.

---

## 📞 Farmer Support Resources

| Resource | Details |
|---|---|
| 🇮🇳 Kisan Call Centre | **1800-180-1551** (Toll Free, 24x7) |
| 🏫 KVK (Krishi Vigyan Kendra) | Find your nearest center at [kvk.icar.gov.in](https://kvk.icar.gov.in) |
| 🌐 Farmer Portal | [farmer.gov.in](https://farmer.gov.in) |
| 📱 mKisan | Free SMS advisory for registered farmers |

---

## 👩‍💻 Made by

**Akanksha Kadam**

---

## 📄 License

This project is for educational and research purposes. The PlantVillage dataset is used under its original license terms.
