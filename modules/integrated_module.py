"""
RoadSafe AI - Integrated Multi-Modal Risk Analysis Module (Module D)
Fuses Computer Vision traffic density, Machine Learning predictive risk,
and Natural Language incident intelligence into a unified decision-support assessment.
"""

import datetime
import plotly.graph_objects as go


def compute_integrated_risk(cv_result=None, ml_result=None, nlp_result=None):
    """
    Synthesizes multi-modal signals into an overarching composite risk rating.
    """
    scores = []
    weights = []
    modalities_used = []
    
    # ML Component (Weight: 45%)
    if ml_result is not None:
        ml_score = ml_result.get('risk_index', 50.0)
        scores.append(ml_score)
        weights.append(0.45)
        modalities_used.append("Machine Learning Environmental Classifier")
    else:
        ml_score = None
        
    # CV Component (Weight: 30%)
    if cv_result is not None:
        cv_score = cv_result.get('density_score', 50.0)
        # If pedestrians are present, boost CV risk
        ped_count = cv_result.get('pedestrians', 0)
        if ped_count > 0:
            cv_score = min(100.0, cv_score + ped_count * 8.0)
        scores.append(cv_score)
        weights.append(0.30)
        modalities_used.append("Computer Vision & YOLO Object Detection")
    else:
        cv_score = None
        
    # NLP Component (Weight: 25%)
    if nlp_result is not None:
        nlp_score = float(nlp_result.get('urgency_score', 30.0))
        scores.append(nlp_score)
        weights.append(0.25)
        modalities_used.append("Natural Language Narrative Extractor")
    else:
        nlp_score = None
        
    if not scores:
        return {
            'composite_score': 0.0,
            'composite_level': 'Low',
            'recommendations': ['Please execute at least one AI module to generate recommendations.'],
            'modalities_used': []
        }
        
    # Normalized weighted average
    total_weight = sum(weights)
    composite_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    composite_score = round(float(composite_score), 1)
    
    if composite_score >= 70.0:
        composite_level = 'High'
        status_color = '#EF4444'
        badge = '🔴 HIGH RISK — CRITICAL ALERT'
    elif composite_score >= 38.0:
        composite_level = 'Medium'
        status_color = '#F59E0B'
        badge = '🟡 MEDIUM RISK — ELEVATED CAUTION'
    else:
        composite_level = 'Low'
        status_color = '#10B981'
        badge = '🟢 LOW RISK — NORMAL CONDITIONS'
        
    # Generate proactive safety recommendations
    recommendations = []
    
    if composite_level == 'High':
        recommendations.append("🚨 **Emergency Readiness:** Alert highway patrol and municipal traffic operations center for active incident monitoring.")
        recommendations.append("⚠️ **Variable Speed Limits (VSL):** Dynamically reduce posted speed limit by 15-20 mph on digital gantries.")
        recommendations.append("📢 **VMS Advisory Boards:** Broadcast warning alerts: *'Severe Road Hazard Ahead - Exercise Extreme Caution'*.")
    elif composite_level == 'Medium':
        recommendations.append("⚡ **Caution Advisory:** Activate roadside advisory flashers and overhead safety warning signs.")
        recommendations.append("🚗 **Traffic Spacing:** Recommend increasing minimum vehicle following distance from 2 seconds to 4-5 seconds.")
    else:
        recommendations.append("✅ **Standard Monitoring:** Routine surveillance active; no critical safety interventions required.")

    # Context-specific safety measures
    if cv_result and cv_result.get('pedestrians', 0) > 0:
        recommendations.append(f"🚶 **Pedestrian Protection Protocol:** High vulnerability zone ({cv_result.get('pedestrians')} pedestrians detected). Prioritize crosswalk signal timing.")
        
    if ml_result and any('Hazardous Surface' in f or 'Adverse Weather' in f for f in ml_result.get('contributing_factors', [])):
        recommendations.append("🚜 **Road Maintenance Dispatch:** Request salting / de-icing / drainage clearing units to mitigate hazardous surface friction.")
        
    if nlp_result and any('Critical' in c for c in nlp_result.get('severity_cues', [])):
        recommendations.append("🚑 **Priority EMS Dispatch:** Textual narrative indicates critical vehicle damage / potential occupant injury.")

    return {
        'composite_score': composite_score,
        'composite_level': composite_level,
        'status_color': status_color,
        'badge': badge,
        'ml_score': ml_score,
        'cv_score': cv_score,
        'nlp_score': nlp_score,
        'modalities_used': modalities_used,
        'recommendations': recommendations,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def create_modal_radar_chart(ml_score, cv_score, nlp_score):
    """Creates a multi-axis Radar / Polar chart comparing the 3 AI modules."""
    categories = ['ML Environmental Risk', 'CV Traffic Density', 'NLP Narrative Urgency']
    values = [
        ml_score if ml_score is not None else 0.0,
        cv_score if cv_score is not None else 0.0,
        nlp_score if nlp_score is not None else 0.0
    ]
    
    # Close polygon
    categories_plot = categories + [categories[0]]
    values_plot = values + [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values_plot,
        theta=categories_plot,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38BDF8', width=2),
        marker=dict(size=8, color='#0284C7')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#94A3B8'),
                gridcolor='#334155'
            ),
            angularaxis=dict(
                tickfont=dict(color='#E2E8F0', size=12),
                gridcolor='#334155'
            ),
            bgcolor='rgba(15, 23, 42, 0.6)'
        ),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def generate_markdown_audit_report(summary):
    """Generates a downloadable markdown executive audit report."""
    md = f"""# RoadSafe AI — Integrated Multi-Modal Audit Report
**Generated At:** {summary.get('timestamp', 'N/A')}  
**Overall Risk Status:** {summary.get('badge', 'N/A')}  
**Composite Safety Risk Index:** {summary.get('composite_score', 0)} / 100  

---

## 1. Multi-Modal Intelligence Breakdown
- **Machine Learning Environmental Risk:** {summary.get('ml_score', 'N/A')}/100
- **Computer Vision Density & Road Object Score:** {summary.get('cv_score', 'N/A')}/100
- **NLP Narrative Severity Score:** {summary.get('nlp_score', 'N/A')}/100

## 2. Integrated Decision-Support Protocols
"""
    for rec in summary.get('recommendations', []):
        clean_rec = rec.replace('**', '').replace('🚨 ', '').replace('⚠️ ', '').replace('📢 ', '').replace('⚡ ', '').replace('🚗 ', '').replace('✅ ', '').replace('🚶 ', '').replace('🚜 ', '').replace('🚑 ', '')
        md += f"- {clean_rec}\n"

    md += """
---
*Notice: This assessment is generated by RoadSafe AI academic decision-support prototype combining Machine Learning, Deep Learning (YOLOv8), and Natural Language Processing.*
"""
    return md
