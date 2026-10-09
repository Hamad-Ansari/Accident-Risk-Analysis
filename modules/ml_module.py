"""
RoadSafe AI - Machine Learning Risk Prediction Module (Module B)
Evaluates structured road, weather, and environmental factors to classify
road safety risk into Low, Medium, or High categories.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_risk_model.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "risk_label_encoder.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

_cached_model = None
_cached_encoder = None
_cached_metadata = None


def load_ml_artifacts():
    """Loads the trained ML risk classification pipeline and metadata."""
    global _cached_model, _cached_encoder, _cached_metadata
    
    if _cached_model is not None and _cached_encoder is not None:
        return _cached_model, _cached_encoder, _cached_metadata
        
    try:
        if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
            from train_model import train_and_save_models
            train_and_save_models()
            
        _cached_model = joblib.load(MODEL_PATH)
        _cached_encoder = joblib.load(ENCODER_PATH)
        
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {}
            
        return _cached_model, _cached_encoder, _cached_metadata
    except Exception as e:
        print(f"Error loading ML model artifacts: {e}. Re-training fallback...")
        from train_model import train_and_save_models
        train_and_save_models()
        _cached_model = joblib.load(MODEL_PATH)
        _cached_encoder = joblib.load(ENCODER_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            _cached_metadata = json.load(f)
        return _cached_model, _cached_encoder, _cached_metadata


def get_preset_scenarios():
    """Returns curated realistic environmental scenarios for quick testing."""
    return {
        "⛈️ Severe Night Rainstorm (High Risk)": {
            'Weather_Condition': 'Severe Storm',
            'Road_Condition': 'Standing Water',
            'Time_of_Day': 'Night/Late',
            'Sunrise_Sunset': 'Night',
            'Visibility(mi)': 1.2,
            'Temperature(F)': 48.0,
            'Humidity(%)': 95.0,
            'Wind_Speed(mph)': 34.0,
            'Speed_Limit': 65,
            'Traffic_Signal': 1,
            'Junction': 1,
            'Crossing': 0
        },
        "🌫️ Dense Foggy Morning Junction (High Risk)": {
            'Weather_Condition': 'Fog',
            'Road_Condition': 'Wet',
            'Time_of_Day': 'Morning Rush',
            'Sunrise_Sunset': 'Day',
            'Visibility(mi)': 0.8,
            'Temperature(F)': 42.0,
            'Humidity(%)': 98.0,
            'Wind_Speed(mph)': 8.0,
            'Speed_Limit': 55,
            'Traffic_Signal': 1,
            'Junction': 1,
            'Crossing': 1
        },
        "❄️ Icy Winter Highway (High Risk)": {
            'Weather_Condition': 'Snow',
            'Road_Condition': 'Icy',
            'Time_of_Day': 'Night/Late',
            'Sunrise_Sunset': 'Night',
            'Visibility(mi)': 3.0,
            'Temperature(F)': 24.0,
            'Humidity(%)': 80.0,
            'Wind_Speed(mph)': 22.0,
            'Speed_Limit': 70,
            'Traffic_Signal': 0,
            'Junction': 1,
            'Crossing': 0
        },
        "⛅ Moderate Afternoon Commute (Medium Risk)": {
            'Weather_Condition': 'Overcast',
            'Road_Condition': 'Wet',
            'Time_of_Day': 'Evening Rush',
            'Sunrise_Sunset': 'Day',
            'Visibility(mi)': 6.0,
            'Temperature(F)': 60.0,
            'Humidity(%)': 70.0,
            'Wind_Speed(mph)': 12.0,
            'Speed_Limit': 45,
            'Traffic_Signal': 1,
            'Junction': 0,
            'Crossing': 1
        },
        "☀️ Clear Daytime Boulevard (Low Risk)": {
            'Weather_Condition': 'Clear',
            'Road_Condition': 'Dry',
            'Time_of_Day': 'Daytime',
            'Sunrise_Sunset': 'Day',
            'Visibility(mi)': 10.0,
            'Temperature(F)': 72.0,
            'Humidity(%)': 45.0,
            'Wind_Speed(mph)': 5.0,
            'Speed_Limit': 35,
            'Traffic_Signal': 1,
            'Junction': 0,
            'Crossing': 0
        }
    }


def predict_risk(feature_dict):
    """
    Runs model inference on input features and returns risk assessment details.
    """
    model, encoder, metadata = load_ml_artifacts()
    
    # Create single-row DataFrame
    df_input = pd.DataFrame([feature_dict])
    
    # Predict encoded class and probabilities
    pred_encoded = model.predict(df_input)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df_input)[0]
        prob_dict = {str(cls): float(prob) for cls, prob in zip(encoder.classes_, probabilities)}
    else:
        prob_dict = {str(cls): (1.0 if cls == pred_label else 0.0) for cls in encoder.classes_}
        
    # Calculate composite risk index (0 to 100)
    low_prob = prob_dict.get('Low', 0.0)
    med_prob = prob_dict.get('Medium', 0.0)
    high_prob = prob_dict.get('High', 0.0)
    risk_index = round(float(low_prob * 15.0 + med_prob * 55.0 + high_prob * 95.0), 1)
    
    # Analyze contributing risk factors
    contributing_factors = []
    if feature_dict.get('Weather_Condition') in ['Severe Storm', 'Snow', 'Fog', 'Rain']:
        contributing_factors.append(f"Adverse Weather: {feature_dict['Weather_Condition']}")
    if feature_dict.get('Road_Condition') in ['Icy', 'Snow Covered', 'Standing Water', 'Wet']:
        contributing_factors.append(f"Hazardous Surface: {feature_dict['Road_Condition']}")
    if float(feature_dict.get('Visibility(mi)', 10)) < 3.0:
        contributing_factors.append(f"Severely Reduced Visibility ({feature_dict['Visibility(mi)']} mi)")
    if feature_dict.get('Time_of_Day') == 'Night/Late' or feature_dict.get('Sunrise_Sunset') == 'Night':
        contributing_factors.append("Low Light / Night-time Driving")
    if feature_dict.get('Time_of_Day') in ['Morning Rush', 'Evening Rush']:
        contributing_factors.append("Peak Traffic Rush Hour")
    if int(feature_dict.get('Junction', 0)) == 1:
        contributing_factors.append("High-Conflict Road Junction / Interchange")
    if float(feature_dict.get('Wind_Speed(mph)', 0)) > 25.0:
        contributing_factors.append(f"High Gusting Winds ({feature_dict['Wind_Speed(mph)']} mph)")
    if int(feature_dict.get('Speed_Limit', 35)) >= 65:
        contributing_factors.append(f"High-Speed Zone ({feature_dict['Speed_Limit']} mph)")

    if not contributing_factors:
        contributing_factors.append("Optimal road and weather conditions detected")
        
    return {
        'risk_level': pred_label,
        'confidence': prob_dict.get(pred_label, 1.0),
        'probabilities': prob_dict,
        'risk_index': risk_index,
        'contributing_factors': contributing_factors
    }


def create_risk_probability_chart(prob_dict):
    """Generates an interactive Plotly bar chart for risk class probabilities."""
    labels = ['Low', 'Medium', 'High']
    values = [prob_dict.get(k, 0.0) * 100 for k in labels]
    colors = ['#10B981', '#F59E0B', '#EF4444']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title="Class Probability Distribution",
        title_font=dict(size=14, color="#E2E8F0"),
        xaxis=dict(title="Risk Category", color="#94A3B8"),
        yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="#334155", color="#94A3B8"),
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_risk_gauge(risk_index, risk_label):
    """Generates an interactive circular risk index gauge."""
    color_map = {
        'Low': '#10B981',
        'Medium': '#F59E0B',
        'High': '#EF4444'
    }
    color = color_map.get(risk_label, '#3B82F6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overall Risk Level: <b>{risk_label.upper()}</b>", 'font': {'size': 16, 'color': '#E2E8F0'}},
        number={'suffix': "/100", 'font': {'color': color, 'size': 26}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.8,
                'value': risk_index
            }
        }
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
