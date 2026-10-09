"""
RoadSafe AI - NLP Accident Report Analyzer (Module C)
Extracts structured intelligence (accident types, involved vehicles, weather,
road conditions, severity cues) from free-form textual incident narratives.
"""

import re
import plotly.graph_objects as go

# Keyword and regex dictionaries for structured entity extraction
ENTITY_PATTERNS = {
    'accident_types': {
        'Multi-Vehicle Pileup': [r'pile-?up', r'chain reaction', r'multi(ple)?-?vehicle', r'multiple cars', r'series of collisions'],
        'Head-On Collision': [r'head-?on', r'crossed centerline', r'wrong way', r'frontal impact'],
        'Rear-End Collision': [r'rear-?ended?', r'tailgat', r'struck from behind', r'hit the rear'],
        'T-Bone / Side-Impact': [r't-?bone', r'side-?impact', r'broadside', r'perpendicular'],
        'Rollover Crash': [r'roll(ed)?-?over', r'flipped', r'overturned', r'roof impact'],
        'Pedestrian / Cyclist Incident': [r'pedestrian', r'hit a walker', r'cyclist', r'struck a person', r'jaywalk', r'bicycle rider'],
        'Single Vehicle Run-Off': [r'ran off (the )?road', r'hit a guardrail', r'struck a tree', r'lost control', r'slid into ditch', r'hydroplan']
    },
    'vehicles_involved': {
        'Commercial Truck / Semi': [r'semi-?truck', r'tractor-?trailer', r'big rig', r'18-?wheeler', r'freight truck', r'heavy truck', r'tanker'],
        'Passenger Car / Sedan': [r'car', r'sedan', r'coupe', r'hatchback', r'vehicle', r'automobile'],
        'SUV / Van / Pickup': [r'suv', r'pickup', r'van', r'minivan', r'truck', r'crossover'],
        'Motorcycle': [r'motorcycle', r'motorbike', r'biker', r'scooter', r'two-wheeler'],
        'Bicycle': [r'bicycle', r'bike', r'cyclist'],
        'Public Transit Bus': [r'bus', r'transit', r'coach', r'school bus'],
        'Emergency Vehicle': [r'ambulance', r'police car', r'fire truck']
    },
    'weather_conditions': {
        'Heavy Rain / Downpour': [r'heavy rain', r'downpour', r'torrential', r'pouring', r'storm', r'monsoon'],
        'Rain / Drizzle': [r'rain', r'drizzle', r'wet weather', r'showers'],
        'Snow / Blizzard': [r'snow', r'blizzard', r'sleet', r'snowfall', r'hail'],
        'Dense Fog / Low Visibility': [r'dense fog', r'thick fog', r'mist', r'hazy', r'zero visibility', r'smog'],
        'Clear / Sunny': [r'clear sky', r'sunny', r'dry weather', r'good weather', r'bright daylight'],
        'Overcast / Cloudy': [r'overcast', r'cloudy', r'gloomy']
    },
    'road_conditions': {
        'Icy / Black Ice': [r'black ice', r'icy', r'freezing road', r'frost', r'frozen'],
        'Wet / Slippery': [r'wet road', r'slippery', r'slick', r'water puddles'],
        'Flooded / Standing Water': [r'flooded', r'standing water', r'hydroplaning water', r'submerged'],
        'Snow-Covered': [r'snow-covered', r'slush', r'unplowed'],
        'Damaged / Construction': [r'pothole', r'construction zone', r'debris', r'road work', r'gravel'],
        'Dry / Normal': [r'dry road', r'dry asphalt', r'normal pavement']
    },
    'lighting_time': {
        'Night / Dark': [r'night', r'dark', r'midnight', r'unlit road', r'after dark', r'1[0-2] pm', r'[7-9] pm'],
        'Dusk / Twilight': [r'dusk', r'twilight', r'sunset', r'evening'],
        'Dawn / Early Morning': [r'dawn', r'daybreak', r'early morning', r'[4-6] am'],
        'Daylight': [r'daytime', r'broad daylight', r'afternoon', r'noon', r'morning']
    },
    'severity_cues': {
        'Critical / Severe': [r'fatal', r'casualt(y|ies)', r'critical condition', r'trapped in vehicle', r'airlift', r'extricat', r'unconscious', r'severe injur', r'totaled'],
        'Moderate Injury': [r'injured', r'hospital', r'paramedics', r'ambulance dispatched', r'broken', r'concussion', r'airbags? deployed'],
        'Minor / Property Damage Only': [r'minor damage', r'fender bender', r'scratched', r'no injuries', r'minor cuts', r'refused medical attention']
    }
}


def get_sample_accident_reports():
    """Returns realistic accident reports for interactive demoing."""
    return {
        "🚨 Severe Multi-Vehicle Highway Pileup": (
            "At approximately 9:45 PM on a dark and unlit stretch of Interstate 80, a severe multi-vehicle "
            "pileup occurred during heavy rain and standing water. A speeding commercial semi-truck hydroplaned, "
            "losing control and jackknifing across two lanes. Three passenger sedans and an SUV rear-ended the truck in a chain reaction. "
            "Airbags deployed in all vehicles and multiple occupants suffered critical injuries requiring emergency extrication."
        ),
        "🌫️ Morning Fog Intersection & Pedestrian Incident": (
            "During the morning commute around 7:15 AM, dense fog caused zero visibility near the downtown junction. "
            "A delivery van ran a red traffic signal and struck a pedestrian attempting to cross the intersection. "
            "Paramedics were immediately dispatched to treat moderate leg fractures."
        ),
        "❄️ Icy Bridge Rollover": (
            "A pickup truck crossed an unplowed bridge at 11:30 PM and encountered black ice. The vehicle spun 180 degrees, "
            "struck the concrete guardrail, and overturned into the embankment. The driver sustained minor cuts with significant vehicle damage."
        ),
        "☀️ Clear Afternoon Low-Speed Fender Bender": (
            "In broad daylight on a sunny afternoon, a sedan was waiting at a red traffic signal on Main Street. "
            "A following hatchback failed to brake in time and caused a minor low-speed rear-end collision. "
            "Both drivers exchanged insurance information with property damage only and no injuries reported."
        )
    }


def extract_accident_info(text):
    """
    Parses unstructured accident narrative into rich structured dictionary.
    """
    if not text or not text.strip():
        return {
            'accident_types': ['Unspecified Incident'],
            'vehicles_involved': ['Unspecified Vehicle'],
            'weather_conditions': ['Not Mentioned'],
            'road_conditions': ['Not Mentioned'],
            'lighting_time': ['Not Mentioned'],
            'severity_indicators': ['Unspecified'],
            'risk_level': 'Low',
            'urgency_score': 20,
            'summary': 'No text provided'
        }
        
    text_lower = text.lower()
    extracted = {}
    
    for category, subcategories in ENTITY_PATTERNS.items():
        found = []
        for label, patterns in subcategories.items():
            for p in patterns:
                if re.search(p, text_lower):
                    found.append(label)
                    break
        extracted[category] = found if found else ['Not Explicitly Mentioned']
        
    # Urgency & Risk Scoring heuristics
    urgency = 25
    
    # Severity keywords
    if any(k in extracted['severity_cues'] for k in ['Critical / Severe']):
        urgency += 45
    elif any(k in extracted['severity_cues'] for k in ['Moderate Injury']):
        urgency += 25
    elif any(k in extracted['severity_cues'] for k in ['Minor / Property Damage Only']):
        urgency += 5
        
    # Weather & Road Hazard keywords
    if any(k in extracted['weather_conditions'] for k in ['Heavy Rain / Downpour', 'Snow / Blizzard', 'Dense Fog / Low Visibility']):
        urgency += 15
    if any(k in extracted['road_conditions'] for k in ['Icy / Black Ice', 'Flooded / Standing Water', 'Wet / Slippery']):
        urgency += 15
        
    # Multi-vehicle or Rollover
    if any(k in extracted['accident_types'] for k in ['Multi-Vehicle Pileup', 'Rollover Crash', 'Head-On Collision']):
        urgency += 20
        
    urgency = min(100, max(10, urgency))
    
    if urgency >= 70:
        predicted_risk = 'High'
    elif urgency >= 40:
        predicted_risk = 'Medium'
    else:
        predicted_risk = 'Low'
        
    return {
        'accident_types': extracted['accident_types'],
        'vehicles_involved': extracted['vehicles_involved'],
        'weather_conditions': extracted['weather_conditions'],
        'road_conditions': extracted['road_conditions'],
        'lighting_time': extracted['lighting_time'],
        'severity_cues': extracted['severity_cues'],
        'risk_level': predicted_risk,
        'urgency_score': urgency,
        'word_count': len(text.split()),
        'character_count': len(text)
    }


def create_urgency_gauge(urgency_score, risk_label):
    """Generates an interactive Plotly gauge for NLP Urgency/Severity score."""
    color_map = {
        'Low': '#10B981',
        'Medium': '#F59E0B',
        'High': '#EF4444'
    }
    color = color_map.get(risk_label, '#3B82F6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=urgency_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"NLP Incident Severity: <b>{risk_label.upper()}</b>", 'font': {'size': 16, 'color': '#E2E8F0'}},
        number={'suffix': "/100", 'font': {'color': color, 'size': 26}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 39], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [39, 69], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [69, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ]
        }
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
