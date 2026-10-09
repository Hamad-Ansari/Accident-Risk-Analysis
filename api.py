"""
RoadSafe AI — FastAPI Enterprise Microservice (api.py)
Provides high-performance REST API endpoints for ML Risk Prediction,
YOLOv8 Computer Vision Object Detection, NLP Report Analysis, and Multi-Modal Integration.
"""

import io
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

# Import AI Engines
from modules.ml_module import predict_risk, load_ml_artifacts
from modules.cv_module import detect_road_objects
from modules.nlp_module import extract_accident_info
from modules.integrated_module import compute_integrated_risk

# Initialize FastAPI App with rich Swagger OpenAPI documentation
app = FastAPI(
    title="RoadSafe AI — REST API Microservice",
    description="""
    **Enterprise AI Decision-Support API for Intelligent Road Safety & Accident Risk Analysis.**
    
    Integrates 4 Core Modules:
    - 🏎️ **Module A (Computer Vision):** YOLOv8 Road Object Detection & Congestion Estimation
    - ⚡ **Module B (Machine Learning):** Environmental Accident Risk Classification
    - 📝 **Module C (NLP):** Incident Narrative Entity & Severity Extraction
    - 🌐 **Module D (Multi-Modal):** Unified Cross-Discipline Safety Audit & Protocols
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web/mobile client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PYDANTIC DATA SCHEMAS
# ==========================================

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    timestamp: float
    services: Dict[str, str]


class RiskPredictionRequest(BaseModel):
    weather_condition: str = Field(default="Rain", description="Weather condition e.g. Clear, Rain, Fog, Snow, Severe Storm")
    road_condition: str = Field(default="Wet", description="Road surface condition e.g. Dry, Wet, Icy, Standing Water")
    time_of_day: str = Field(default="Night/Late", description="Time of day e.g. Morning Rush, Daytime, Evening Rush, Night/Late")
    visibility_mi: float = Field(default=2.5, ge=0.1, le=10.0, description="Visibility in miles (0.1 to 10.0)")
    temperature_f: float = Field(default=45.0, description="Ambient temperature in Fahrenheit")
    humidity_pct: float = Field(default=85.0, ge=0.0, le=100.0, description="Relative humidity percentage")
    wind_speed_mph: float = Field(default=18.0, ge=0.0, description="Wind speed in mph")
    speed_limit: int = Field(default=55, description="Posted road speed limit in mph")
    traffic_signal: int = Field(default=1, ge=0, le=1, description="1 if traffic signal is present, else 0")
    junction: int = Field(default=1, ge=0, le=1, description="1 if road junction/interchange is present, else 0")
    crossing: int = Field(default=0, ge=0, le=1, description="1 if pedestrian crossing is present, else 0")


class RiskPredictionResponse(BaseModel):
    risk_level: str
    risk_index: float
    confidence: float
    probabilities: Dict[str, float]
    contributing_factors: List[str]


class DetectionItem(BaseModel):
    category: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    total_vehicles: int
    pedestrians: int
    total_objects: int
    density_level: str
    density_score: float
    counts_by_class: Dict[str, int]
    detections: List[DetectionItem]


class NLPReportRequest(BaseModel):
    report_text: str = Field(
        ..., 
        min_length=5,
        description="Free-form textual description or narrative of the road accident",
        example="A commercial semi-truck jackknifed in heavy rain on Interstate 80 at 9 PM. Three passenger cars rear-ended the truck, with severe injuries reported."
    )


class NLPReportResponse(BaseModel):
    accident_types: List[str]
    vehicles_involved: List[str]
    weather_conditions: List[str]
    road_conditions: List[str]
    lighting_time: List[str]
    severity_cues: List[str]
    predicted_risk_level: str
    urgency_score: int
    word_count: int


class IntegratedRiskRequest(BaseModel):
    ml_features: Optional[RiskPredictionRequest] = None
    report_text: Optional[str] = None


class IntegratedRiskResponse(BaseModel):
    composite_risk_score: float
    composite_risk_level: str
    status_badge: str
    ml_score: Optional[float]
    cv_score: Optional[float]
    nlp_score: Optional[float]
    actionable_protocols: List[str]
    modalities_used: List[str]
    timestamp: str


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", response_model=HealthCheckResponse, tags=["System Health"])
def root():
    """Returns API health status, available microservices, and system timestamp."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "services": {
            "cv_detector": "YOLOv8 Object Detection & Traffic Density Engine",
            "ml_risk_classifier": "Random Forest Multi-Class Risk Model",
            "nlp_analyzer": "Contextual Accident Narrative Parser",
            "integrated_engine": "Multi-Modal Safety Decision Support Engine"
        }
    }


@app.post("/predict-risk", response_model=RiskPredictionResponse, tags=["Module B: Machine Learning"])
def api_predict_risk(payload: RiskPredictionRequest):
    """
    Evaluates environmental and road infrastructure features to predict accident risk (Low, Medium, High).
    """
    try:
        sunrise_sunset = "Night" if payload.time_of_day == "Night/Late" else "Day"
        
        feature_dict = {
            'Weather_Condition': payload.weather_condition,
            'Road_Condition': payload.road_condition,
            'Time_of_Day': payload.time_of_day,
            'Sunrise_Sunset': sunrise_sunset,
            'Visibility(mi)': payload.visibility_mi,
            'Temperature(F)': payload.temperature_f,
            'Humidity(%)': payload.humidity_pct,
            'Wind_Speed(mph)': payload.wind_speed_mph,
            'Speed_Limit': payload.speed_limit,
            'Traffic_Signal': payload.traffic_signal,
            'Junction': payload.junction,
            'Crossing': payload.crossing
        }
        
        res = predict_risk(feature_dict)
        return {
            "risk_level": res["risk_level"],
            "risk_index": res["risk_index"],
            "confidence": round(res["confidence"], 4),
            "probabilities": res["probabilities"],
            "contributing_factors": res["contributing_factors"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Risk Prediction error: {str(e)}")


@app.post("/detect-objects", response_model=DetectionResponse, tags=["Module A: Computer Vision"])
async def api_detect_objects(
    file: UploadFile = File(..., description="Upload road image or frame (JPG/PNG)"),
    conf_threshold: float = Query(0.35, ge=0.1, le=0.9, description="Confidence threshold")
):
    """
    Runs YOLOv8 object detection on an uploaded image, counting road objects and estimating traffic density.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        res = detect_road_objects(image, conf_threshold=conf_threshold)
        
        detections = [
            DetectionItem(
                category=d["class"],
                confidence=round(d["confidence"], 4),
                bbox=[round(x, 2) for x in d["bbox"]]
            )
            for d in res["raw_detections"]
        ]
        
        return {
            "total_vehicles": res["total_vehicles"],
            "pedestrians": res["pedestrians"],
            "total_objects": res["total_objects"],
            "density_level": res["density"],
            "density_score": res["density_score"],
            "counts_by_class": res["counts"],
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computer Vision detection error: {str(e)}")


@app.post("/analyze-report", response_model=NLPReportResponse, tags=["Module C: Natural Language Processing"])
def api_analyze_report(payload: NLPReportRequest):
    """
    Parses unstructured accident descriptions to extract involved vehicle types, weather hazards, and urgency levels.
    """
    try:
        res = extract_accident_info(payload.report_text)
        return {
            "accident_types": res["accident_types"],
            "vehicles_involved": res["vehicles_involved"],
            "weather_conditions": res["weather_conditions"],
            "road_conditions": res["road_conditions"],
            "lighting_time": res["lighting_time"],
            "severity_cues": res["severity_cues"],
            "predicted_risk_level": res["risk_level"],
            "urgency_score": res["urgency_score"],
            "word_count": res["word_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP Report Analysis error: {str(e)}")


@app.post("/integrated-risk", response_model=IntegratedRiskResponse, tags=["Module D: Integrated Multi-Modal"])
def api_integrated_risk(payload: IntegratedRiskRequest):
    """
    Synthesizes multi-modal inputs into an overarching composite risk score and generates action protocols.
    """
    try:
        ml_res = None
        if payload.ml_features is not None:
            f = payload.ml_features
            sunrise_sunset = "Night" if f.time_of_day == "Night/Late" else "Day"
            feature_dict = {
                'Weather_Condition': f.weather_condition,
                'Road_Condition': f.road_condition,
                'Time_of_Day': f.time_of_day,
                'Sunrise_Sunset': sunrise_sunset,
                'Visibility(mi)': f.visibility_mi,
                'Temperature(F)': f.temperature_f,
                'Humidity(%)': f.humidity_pct,
                'Wind_Speed(mph)': f.wind_speed_mph,
                'Speed_Limit': f.speed_limit,
                'Traffic_Signal': f.traffic_signal,
                'Junction': f.junction,
                'Crossing': f.crossing
            }
            ml_res = predict_risk(feature_dict)
            
        nlp_res = None
        if payload.report_text is not None and payload.report_text.strip():
            nlp_res = extract_accident_info(payload.report_text)
            
        summary = compute_integrated_risk(ml_result=ml_res, nlp_result=nlp_res)
        
        return {
            "composite_risk_score": summary["composite_score"],
            "composite_risk_level": summary["composite_level"],
            "status_badge": summary["badge"],
            "ml_score": summary["ml_score"],
            "cv_score": summary["cv_score"],
            "nlp_score": summary["nlp_score"],
            "actionable_protocols": summary["recommendations"],
            "modalities_used": summary["modalities_used"],
            "timestamp": summary["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integrated Risk Analysis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting RoadSafe AI FastAPI Microservice on http://127.0.0.1:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
