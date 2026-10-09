# 🚦 RoadSafe AI — Enterprise Road Safety & Accident Risk Analysis

**RoadSafe AI** is an end-to-end multi-modal Artificial Intelligence platform that unifies **Computer Vision (YOLOv8 Dashcam Video & Image Stream)**, **Machine Learning Risk Prediction**, and **Natural Language Processing (NLP)** to analyze traffic density timelines, evaluate environmental accident risk, and extract structured intelligence from textual incident narratives.

Includes both an interactive **Streamlit Web Application** and a high-performance **FastAPI REST Microservice**.

---

## 🌟 Key Features & Enterprise Modules

### 1. 📊 Interactive Executive Dashboard
- Unified view of road safety indicators, KPI metrics, active AI disciplines, and system architecture.

### 2. 🚗 Module A: Traffic Vision (Real-Time Image & Dashcam Video Analytics)
- **Model:** Pretrained YOLOv8 Object Detector (`yolov8n`).
- **Entity Classes:** Cars, Trucks, Buses, Motorcycles, Bicycles, and Pedestrians.
- **Video & Dashcam Ingestion:** Upload `.mp4`, `.avi`, `.mov` footage or live camera frames.
- **Time-Series Density Tracking:** Computes frame-by-frame congestion timelines, peak traffic timestamps, and inference FPS metrics.
- **Visuals:** Real-time bounding boxes, interactive density gauges, and Plotly timeline graphs.

### 3. ⚠️ Module B: Accident Risk (Machine Learning Classifier)
- **Models Evaluated:** Logistic Regression, Random Forest (Best: 82.6% Accuracy, 0.826 F1), and XGBoost.
- **Features Analyzed:** Weather Condition, Road Surface (Dry/Wet/Icy), Visibility (miles), Temperature, Humidity, Wind Speed, Speed Limits, Traffic Signals, and Junctions.
- **Output:** Classifies conditions into **Low Risk 🟢, Medium Risk 🟡, or High Risk 🔴** with class probability distribution and primary contributing hazard factors.

### 4. 📝 Module C: NLP Accident Report Analyzer
- **Technique:** Context-Aware Regex & Named Entity Extraction.
- **Extracted Dimensions:** Collision Types (Rear-end, Head-on, T-Bone, Pileup, Rollover), Vehicles Involved, Road Hazards, Weather Conditions, and Severity/Injury Cues.
- **Scoring:** Computes an incident severity & urgency score ($0 - 100$).

### 5. 🌐 Module D: Integrated Multi-Modal Risk Analysis
- Cross-modal fusion combining Computer Vision density + ML environmental risk + NLP incident severity.
- Computes a unified **Composite Safety Risk Index ($0 - 100$)**.
- Generates actionable safety protocols (Variable Speed Limits, Emergency EMS dispatch alerts, Variable Message Signage).
- Downloadable Executive Audit Reports in Markdown and JSON formats.

### 6. 🔌 FastAPI REST Microservice (`api.py`)
- High-performance asynchronous REST API with automatic Swagger OpenAPI docs (`/docs`).
- Endpoints:
  - `GET /` — Health check & component status
  - `POST /predict-risk` — ML risk classifier
  - `POST /detect-objects` — YOLOv8 vision object detection & density
  - `POST /analyze-report` — NLP narrative parsing
  - `POST /integrated-risk` — Multi-modal synthesis

---

## 🚀 Quick Start (Local Setup)

### 1. Clone Project
```bash
git clone https://github.com/your-username/roadsafe-ai.git
cd "Road safe ai"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run the FastAPI REST Microservice
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open **Interactive Swagger API Docs:** `http://127.0.0.1:8000/docs`

---


## 📁 Project Structure

```
Road safe ai/
├── .streamlit/
│   └── config.toml               # Modern dark theme configuration
├── artifacts/
│   ├── best_risk_model.joblib    # Trained Random Forest ML pipeline
│   ├── risk_label_encoder.joblib # Target label encoder (Low, Medium, High)
│   └── model_metadata.json       # Precomputed metrics and confusion matrix
├── modules/
│   ├── __init__.py
│   ├── cv_module.py              # YOLOv8 object detection & Dashcam video analytics
│   ├── ml_module.py              # ML risk prediction & probability analysis
│   ├── nlp_module.py             # NLP text parser & entity extraction
│   └── integrated_module.py      # Multi-modal fusion & executive reporting
├── sample_images/
│   └── traffic_scene_1.jpg       # Sample road image for immediate testing
├── app.py                        # Main Streamlit web application
├── api.py                        # FastAPI Enterprise REST API microservice
├── train_model.py                # Standalone model training and artifact builder
├── requirements.txt              # Python dependencies
├── packages.txt                  # Linux apt dependencies for cloud hosting
└── README.md                     # Project documentation & deployment guide
```
# Accident-Risk-Analysis
