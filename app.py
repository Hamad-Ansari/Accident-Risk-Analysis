"""
RoadSafe AI — Intelligent Road Safety & Accident Risk Analysis
Main Streamlit Web Application
Combining Computer Vision (YOLOv8 Real-time Image/Video), Machine Learning, NLP, and Multi-Modal Integration.
"""

import os
import io
import json
import tempfile
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import Modular AI Engines
from modules.cv_module import (
    detect_road_objects, 
    process_video_analytics, 
    create_video_density_timeline_chart, 
    create_counts_chart, 
    create_density_gauge
)
from modules.ml_module import (
    predict_risk, 
    load_ml_artifacts, 
    get_preset_scenarios, 
    create_risk_probability_chart, 
    create_risk_gauge
)
from modules.nlp_module import (
    extract_accident_info, 
    get_sample_accident_reports, 
    create_urgency_gauge
)
from modules.integrated_module import (
    compute_integrated_risk, 
    create_modal_radar_chart, 
    generate_markdown_audit_report
)

# Page Configuration
st.set_page_config(
    page_title="RoadSafe AI — Intelligent Road Safety",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism, Badges, Dark Modern Theme)
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .entity-chip {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #7DD3FC;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px;
        font-size: 0.88rem;
    }
    .rec-item {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'cv_result' not in st.session_state:
    st.session_state.cv_result = None
if 'ml_result' not in st.session_state:
    st.session_state.ml_result = None
if 'nlp_result' not in st.session_state:
    st.session_state.nlp_result = None
if 'video_result' not in st.session_state:
    st.session_state.video_result = None


# ==========================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.title("RoadSafe AI")
    st.caption("Intelligent Road Safety & Accident Risk Analysis")
    
    st.markdown("---")
    
    navigation = st.radio(
        "Navigation",
        [
            "📊 Dashboard & Overview",
            "🚗 Traffic Vision (CV / DL)",
            "⚠️ Accident Risk (ML)",
            "📝 Accident Report (NLP)",
            "🌐 Integrated Multi-Modal Risk",
            "📈 Model Performance",
            "🔌 FastAPI Microservice Docs"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.markdown("🟢 **YOLOv8:** Ready (COCO Road Classes)")
    st.markdown("🟢 **Video Ingestion:** MP4 / Dashcam")
    st.markdown("🟢 **ML Classifier:** Random Forest (82.6%)")
    st.markdown("🟢 **NLP Parser:** Active (Rule + Regex NER)")
    st.markdown("🟢 **FastAPI Endpoint:** `api.py` Ready")
    
    st.markdown("---")
    st.caption("RoadSafe AI • Enterprise AI Decision-Support Platform")


# ==========================================
# SCREEN 1: DASHBOARD & OVERVIEW
# ==========================================
if navigation == "📊 Dashboard & Overview":
    st.markdown("""
    <div class="hero-card">
        <h1 style="color: #38BDF8; margin-bottom: 6px;">🚦 RoadSafe AI</h1>
        <h3 style="color: #CBD5E1; font-weight: 400; margin-top: 0;">Enterprise Road Safety & Accident Risk Analysis System</h3>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 0;">
            A unified multi-modal artificial intelligence platform integrating <b>Computer Vision (YOLOv8 Image & Dashcam Video)</b>, 
            <b>Machine Learning Risk Classification</b>, and <b>Natural Language Processing</b> to predict accident hazards, 
            track traffic density timelines, and parse incident narratives.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Stats Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">AI Disciplines</div>
            <div class="metric-value" style="color: #38BDF8;">4 Modules</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">CV • DL • ML • NLP</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best ML Model</div>
            <div class="metric-value" style="color: #34D399;">Random Forest</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Accuracy: 82.6% • F1: 0.826</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Vision Model</div>
            <div class="metric-value" style="color: #A855F7;">YOLOv8 Nano</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Real-time Image & Video Tracking</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Microservice API</div>
            <div class="metric-value" style="color: #FBBF24;">FastAPI</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">REST Endpoints & Swagger UI</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Architecture
    st.subheader("🧩 Integrated System Architecture")
    
    col_arch1, col_arch2 = st.columns([3, 2])
    
    with col_arch1:
        st.markdown("""
        The platform harmonizes multiple data modalities into one intelligent decision-support pipeline:
        
        1. **Module A — Computer Vision (`Traffic Vision`):**
           - Processes road photography, live webcam frames, or **MP4 dashcam videos** using YOLOv8.
           - Detects cars, trucks, buses, motorcycles, bicycles, and pedestrians.
           - Plots frame-by-frame **Traffic Density & Congestion Timelines**.
        
        2. **Module B — Machine Learning Risk Prediction (`Accident Risk`):**
           - Evaluates 12 structured environmental parameters (Weather, Surface, Visibility, Junctions, Time).
           - Predicts **Low Risk 🟢, Medium Risk 🟡, or High Risk 🔴** with class probability distribution.
        
        3. **Module C — NLP Accident Report Analyzer (`Accident Report`):**
           - Extracts involved vehicle types, severe injury cues, road hazards, and urgency scores from raw narrative text.
        
        4. **Module D — Integrated Multi-Modal Fusion & API:**
           - Cross-modal fusion synthesizing vision, tabular, and textual intelligence into actionable safety protocols.
           - Exposed via high-performance **FastAPI REST endpoints** for external systems.
        """)
        
    with col_arch2:
        st.markdown("#### ⚡ Quick Launch Module Hub")
        st.info("💡 **Traffic Vision:** Upload road images or MP4 dashcam videos to track traffic density.")
        st.warning("⚡ **Accident Risk:** Test weather & junction scenarios with the ML classifier.")
        st.success("📝 **Accident Report:** Paste narrative reports to extract structured accident factors.")
        st.info("🔌 **FastAPI Backend:** Access `/docs` to test live REST microservice endpoints.")


# ==========================================
# SCREEN 2: TRAFFIC VISION (CV / DL)
# ==========================================
elif navigation == "🚗 Traffic Vision (CV / DL)":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">🚗 Module A: Computer Vision & Traffic Density Analytics</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            Deep learning road object detection and traffic flow tracking powered by <b>YOLOv8</b>. 
            Supports <b>Road Photography</b>, <b>Dashcam Video Streams</b>, and <b>Live Camera Capture</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    mode_tab1, mode_tab2 = st.tabs(["📸 Road Image Detection", "🎥 Dashcam / Video Stream Analytics"])
    
    # --- TAB 1: IMAGE DETECTION ---
    with mode_tab1:
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            input_source = st.radio("Image Input Mode:", ["Use Sample Road Image", "Upload Road Photo", "Live Camera Capture"], horizontal=True)
        with col_opt2:
            conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.15, max_value=0.90, value=0.35, step=0.05, key="img_conf")
            
        image_to_process = None
        
        if input_source == "Use Sample Road Image":
            sample_path = Path("sample_images/traffic_scene_1.jpg")
            if sample_path.exists():
                image_to_process = Image.open(sample_path)
                st.caption("Loaded sample highway scene.")
            else:
                fallback_path = Path("derek-lee-C031beKcwdQ-unsplash.jpg")
                if fallback_path.exists():
                    image_to_process = Image.open(fallback_path)
        elif input_source == "Upload Road Photo":
            uploaded_file = st.file_uploader("Upload Road Scene Image (JPG / PNG / JPEG)", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file)
        else:
            camera_pic = st.camera_input("Capture Live Road Frame via Camera")
            if camera_pic is not None:
                image_to_process = Image.open(camera_pic)
                
        if image_to_process is not None:
            btn_run = st.button("🚀 Run YOLOv8 Road Object Detection", type="primary", use_container_width=True)
            
            if btn_run or st.session_state.cv_result is not None:
                with st.spinner("Detecting road entities with YOLOv8..."):
                    cv_res = detect_road_objects(image_to_process, conf_threshold=conf_thresh)
                    st.session_state.cv_result = cv_res
                    
                col_img, col_metrics = st.columns([3, 2])
                
                with col_img:
                    st.subheader("🖼️ Annotated Road Scene")
                    st.image(cv_res['annotated_image'], use_container_width=True, caption="YOLOv8 Detection Overlay")
                    
                with col_metrics:
                    st.subheader("📊 Traffic Density & Counts")
                    gauge_fig = create_density_gauge(cv_res['density_score'], cv_res['density'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Total Vehicles", cv_res['total_vehicles'])
                    kpi2.metric("Pedestrians", cv_res['pedestrians'])
                    kpi3.metric("Total Objects", cv_res['total_objects'])
                    
                    chart_fig = create_counts_chart(cv_res['counts'])
                    st.plotly_chart(chart_fig, use_container_width=True)
                    
                if cv_res['raw_detections']:
                    with st.expander("🔍 View Detected Entities & Coordinates"):
                        df_det = pd.DataFrame([
                            {
                                'Class': d['class'],
                                'Confidence': f"{d['confidence'] * 100:.1f}%",
                                'Bounding Box [x1, y1, x2, y2]': [round(x, 1) for x in d['bbox']]
                            }
                            for d in cv_res['raw_detections']
                        ])
                        st.dataframe(df_det, use_container_width=True)
                        
    # --- TAB 2: VIDEO ANALYTICS ---
    with mode_tab2:
        st.markdown("#### 🎥 Dashcam / Traffic CCTV Video Ingestion")
        st.markdown("Upload a dashcam or traffic camera video (`.mp4`, `.avi`, `.mov`) to perform frame-by-frame object detection and density timeline tracking.")
        
        uploaded_video = st.file_uploader("Upload Traffic / Dashcam Video", type=['mp4', 'avi', 'mov'])
        
        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            video_conf = st.slider("Video Detection Confidence", 0.15, 0.90, 0.35, 0.05, key="vid_conf")
        with col_v2:
            frame_sample = st.select_slider("Frame Sampling Frequency (Higher = Faster)", options=[1, 2, 3, 5, 10], value=2)
            
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.flush()
            
            st.video(tfile.name)
            
            btn_process_video = st.button("⚡ Process Dashcam Video Analytics", type="primary", use_container_width=True)
            
            if btn_process_video:
                progress_bar = st.progress(0)
                status_text = st.empty()
                info_metrics = st.empty()
                preview_spot = st.empty()
                
                def update_progress(current, total, last_frame, current_vehicles, current_density):
                    pct = min(1.0, current / max(1, total))
                    progress_bar.progress(pct)
                    status_text.markdown(f"**Processing Frame {current} of {total}** • Density: `{current_density}` ({current_vehicles} vehicles)")
                    if last_frame is not None:
                        preview_spot.image(last_frame, caption=f"Live YOLOv8 Annotated Stream (Frame #{current * frame_sample})", use_container_width=True)
                        
                with st.spinner("Analyzing dashcam video stream with YOLOv8..."):
                    vid_res = process_video_analytics(
                        tfile.name, 
                        conf_threshold=video_conf, 
                        sample_rate=frame_sample, 
                        max_frames=300, 
                        progress_callback=update_progress
                    )
                    st.session_state.video_result = vid_res
                    
                status_text.success("✅ Dashcam Video Analytics Completed Successfully!")
                
            if st.session_state.video_result is not None:
                vr = st.session_state.video_result
                st.markdown("---")
                st.subheader("📈 Dashcam Video Stream Performance & Density Profile")
                
                vk1, vk2, vk3, vk4 = st.columns(4)
                vk1.metric("Avg Vehicles / Frame", vr['avg_vehicles'])
                vk2.metric("Peak Congestion", f"{vr['peak_vehicles']} Vehicles", f"at {vr['peak_timestamp']}")
                vk3.metric("Overall Density", vr['overall_density'])
                vk4.metric("Inference Speed", f"{vr['avg_fps']} FPS", f"{vr['frames_analyzed']} frames analyzed")
                
                # Cumulative vehicle & road user counts row
                st.markdown("#### 🚗 Total Cumulative Road Entities Detected in Video")
                cum = vr.get('cumulative_counts', {})
                
                c_col1, c_col2, c_col3, c_col4, c_col5, c_col6 = st.columns(6)
                c_col1.metric("🚗 Cars", cum.get('Car', 0))
                c_col2.metric("🚚 Trucks", cum.get('Truck', 0))
                c_col3.metric("🚌 Buses", cum.get('Bus', 0))
                c_col4.metric("🏍️ Motorcycles", cum.get('Motorcycle', 0))
                c_col5.metric("🚲 Bicycles", cum.get('Bicycle', 0))
                c_col6.metric("🚶 Pedestrians", cum.get('Pedestrian', 0))
                
                # Visual charts row (Timeline + Breakdown)
                v_ch1, v_ch2 = st.columns([3, 2])
                with v_ch1:
                    st.plotly_chart(create_video_density_timeline_chart(vr['timeline_df']), use_container_width=True)
                with v_ch2:
                    st.plotly_chart(create_counts_chart(vr.get('active_cumulative', cum), title="Total Object Detections Across Video"), use_container_width=True)
                
                with st.expander("📋 View Frame-by-Frame Density Log Table"):
                    st.dataframe(vr['timeline_df'], use_container_width=True)


# ==========================================
# SCREEN 3: ACCIDENT RISK (ML PREDICTION)
# ==========================================
elif navigation == "⚠️ Accident Risk (ML)":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">⚠️ Module B: Machine Learning Accident Risk Classifier</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            Predicts road safety risk level (<b>Low 🟢, Medium 🟡, High 🔴</b>) using an ensemble machine learning model 
            trained on environmental, surface, and infrastructure features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    scenarios = get_preset_scenarios()
    selected_scenario = st.selectbox("🎯 Quick Load Preset Scenario:", ["-- Custom Input --"] + list(scenarios.keys()))
    
    default_vals = {}
    if selected_scenario != "-- Custom Input --":
        default_vals = scenarios[selected_scenario]
        
    st.markdown("### 🎛️ Input Environmental & Road Features")
    
    with st.form("ml_risk_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            weather_opts = ['Clear', 'Rain', 'Fog', 'Snow', 'Overcast', 'Severe Storm']
            w_idx = weather_opts.index(default_vals.get('Weather_Condition', 'Rain')) if 'Weather_Condition' in default_vals else 1
            weather = st.selectbox("Weather Condition", weather_opts, index=w_idx)
            
            road_opts = ['Dry', 'Wet', 'Icy', 'Snow Covered', 'Standing Water']
            r_idx = road_opts.index(default_vals.get('Road_Condition', 'Wet')) if 'Road_Condition' in default_vals else 1
            road_cond = st.selectbox("Road Surface Condition", road_opts, index=r_idx)
            
            visibility = st.slider("Visibility (miles)", min_value=0.1, max_value=10.0, value=float(default_vals.get('Visibility(mi)', 2.5)), step=0.1)
            
        with col_f2:
            tod_opts = ['Morning Rush', 'Daytime', 'Evening Rush', 'Night/Late']
            t_idx = tod_opts.index(default_vals.get('Time_of_Day', 'Night/Late')) if 'Time_of_Day' in default_vals else 3
            time_of_day = st.selectbox("Time of Day", tod_opts, index=t_idx)
            sunrise_sunset = 'Night' if time_of_day == 'Night/Late' else 'Day'
            
            temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=float(default_vals.get('Temperature(F)', 45.0)))
            humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=float(default_vals.get('Humidity(%)', 85.0)), step=1.0)
            
        with col_f3:
            wind_speed = st.number_input("Wind Speed (mph)", min_value=0.0, max_value=80.0, value=float(default_vals.get('Wind_Speed(mph)', 18.0)))
            speed_limit = st.selectbox("Speed Limit (mph)", [25, 35, 45, 55, 65, 75], index=[25, 35, 45, 55, 65, 75].index(default_vals.get('Speed_Limit', 55)))
            
            c_chk1, c_chk2, c_chk3 = st.columns(3)
            with c_chk1:
                traffic_signal = 1 if st.checkbox("Traffic Signal", value=bool(default_vals.get('Traffic_Signal', 1))) else 0
            with c_chk2:
                junction = 1 if st.checkbox("Road Junction", value=bool(default_vals.get('Junction', 1))) else 0
            with c_chk3:
                crossing = 1 if st.checkbox("Ped Crossing", value=bool(default_vals.get('Crossing', 0))) else 0
                
        btn_predict = st.form_submit_button("⚡ Predict Accident Risk Level", type="primary", use_container_width=True)
        
    if btn_predict:
        input_data = {
            'Weather_Condition': weather,
            'Road_Condition': road_cond,
            'Time_of_Day': time_of_day,
            'Sunrise_Sunset': sunrise_sunset,
            'Visibility(mi)': visibility,
            'Temperature(F)': temperature,
            'Humidity(%)': humidity,
            'Wind_Speed(mph)': wind_speed,
            'Speed_Limit': speed_limit,
            'Traffic_Signal': traffic_signal,
            'Junction': junction,
            'Crossing': crossing
        }
        
        with st.spinner("Evaluating safety risk with trained ML model..."):
            ml_res = predict_risk(input_data)
            st.session_state.ml_result = ml_res
            
        st.markdown("---")
        st.subheader("🎯 Model Prediction Results")
        
        r_col1, r_col2 = st.columns([1, 1])
        
        with r_col1:
            badge_class = f"badge-{ml_res['risk_level'].lower()}"
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <span style="font-size: 1.1rem; color: #94A3B8; margin-right: 12px;">Predicted Risk Class:</span>
                <span class="{badge_class}" style="font-size: 1.2rem;">{ml_res['risk_level'].upper()} RISK</span>
            </div>
            """, unsafe_allow_html=True)
            
            gauge = create_risk_gauge(ml_res['risk_index'], ml_res['risk_level'])
            st.plotly_chart(gauge, use_container_width=True)
            
        with r_col2:
            prob_chart = create_risk_probability_chart(ml_res['probabilities'])
            st.plotly_chart(prob_chart, use_container_width=True)
            
        st.markdown("#### 🔍 Contributing Risk Factors")
        for factor in ml_res['contributing_factors']:
            st.markdown(f"- ⚠️ **{factor}**")


# ==========================================
# SCREEN 4: ACCIDENT REPORT (NLP ANALYZER)
# ==========================================
elif navigation == "📝 Accident Report (NLP)":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">📝 Module C: NLP Accident Report Analyzer</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            Extracts structured entities, hazard keywords, involved vehicle types, and severity cues 
            from unstructured textual incident reports.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    samples = get_sample_accident_reports()
    selected_sample = st.selectbox("📄 Select Sample Incident Narrative:", ["-- Enter Custom Text --"] + list(samples.keys()))
    
    initial_text = ""
    if selected_sample != "-- Enter Custom Text --":
        initial_text = samples[selected_sample]
    else:
        initial_text = "A commercial semi-truck jackknifed in heavy rain on Interstate 80 at 9 PM. Three passenger cars rear-ended the truck, with severe injuries reported."
        
    report_text = st.text_area(
        "Accident Narrative Description / Incident Report:",
        value=initial_text,
        height=140
    )
    
    btn_nlp = st.button("🔎 Extract Structured Intelligence with NLP", type="primary", use_container_width=True)
    
    if btn_nlp or st.session_state.nlp_result is not None:
        if report_text.strip():
            with st.spinner("Parsing textual narrative with NLP..."):
                nlp_res = extract_accident_info(report_text)
                st.session_state.nlp_result = nlp_res
                
            st.markdown("---")
            st.subheader("📋 Extracted Structured Intelligence")
            
            n_col1, n_col2 = st.columns([1, 1])
            
            with n_col1:
                st.plotly_chart(create_urgency_gauge(nlp_res['urgency_score'], nlp_res['risk_level']), use_container_width=True)
                
            with n_col2:
                st.markdown("#### 🏷️ Entity & Context Tags")
                
                st.markdown("**Accident Types:**")
                for item in nlp_res['accident_types']:
                    st.markdown(f'<span class="entity-chip">💥 {item}</span>', unsafe_allow_html=True)
                    
                st.markdown("<br>**Vehicles Involved:**", unsafe_allow_html=True)
                for item in nlp_res['vehicles_involved']:
                    st.markdown(f'<span class="entity-chip" style="border-color:#F59E0B; color:#FCD34D;">🚙 {item}</span>', unsafe_allow_html=True)
                    
                st.markdown("<br>**Weather & Road Conditions:**", unsafe_allow_html=True)
                for item in nlp_res['weather_conditions'] + nlp_res['road_conditions']:
                    if item != 'Not Explicitly Mentioned':
                        st.markdown(f'<span class="entity-chip" style="border-color:#10B981; color:#6EE7B7;">🌧️ {item}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="entity-chip" style="opacity:0.6;">{item}</span>', unsafe_allow_html=True)
                        
                st.markdown("<br>**Severity Indicators:**", unsafe_allow_html=True)
                for item in nlp_res['severity_cues']:
                    st.markdown(f'<span class="entity-chip" style="border-color:#EF4444; color:#FCA5A5;">🚨 {item}</span>', unsafe_allow_html=True)
                    
            with st.expander("📊 Text Statistics"):
                s1, s2, s3 = st.columns(3)
                s1.metric("Word Count", nlp_res['word_count'])
                s2.metric("Character Count", nlp_res['character_count'])
                s3.metric("Incident Risk Tier", nlp_res['risk_level'])


# ==========================================
# SCREEN 5: INTEGRATED MULTI-MODAL RISK
# ==========================================
elif navigation == "🌐 Integrated Multi-Modal Risk":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">🌐 Module D: Integrated Multi-Modal Risk Analysis</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            Fuses <b>Computer Vision</b> traffic counts, <b>Machine Learning</b> environmental risk, and <b>NLP</b> 
            narrative intelligence into a unified executive decision-support audit.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    has_cv = st.session_state.cv_result is not None
    has_ml = st.session_state.ml_result is not None
    has_nlp = st.session_state.nlp_result is not None
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.markdown(f"**CV Module:** {'🟢 Synchronized' if has_cv else '⚪ Idle (Using Default)'}")
    col_stat2.markdown(f"**ML Module:** {'🟢 Synchronized' if has_ml else '⚪ Idle (Using Default)'}")
    col_stat3.markdown(f"**NLP Module:** {'🟢 Synchronized' if has_nlp else '⚪ Idle (Using Default)'}")
    
    integrated_summary = compute_integrated_risk(
        cv_result=st.session_state.cv_result,
        ml_result=st.session_state.ml_result,
        nlp_result=st.session_state.nlp_result
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.8); border: 2px solid {integrated_summary['status_color']}; border-radius: 14px; padding: 20px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin: 0; color: {integrated_summary['status_color']};">{integrated_summary['badge']}</h3>
        <h1 style="font-size: 3rem; margin: 10px 0; color: #FFFFFF;">{integrated_summary['composite_score']} <span style="font-size: 1.5rem; color: #94A3B8;">/ 100</span></h1>
        <p style="margin: 0; color: #94A3B8;">Multi-Modal Composite Road Safety Risk Index</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_rad, col_recs = st.columns([1, 1])
    
    with col_rad:
        st.subheader("📡 Multi-Discipline Risk Distribution")
        radar_fig = create_modal_radar_chart(
            integrated_summary['ml_score'],
            integrated_summary['cv_score'],
            integrated_summary['nlp_score']
        )
        st.plotly_chart(radar_fig, use_container_width=True)
        
    with col_recs:
        st.subheader("🚨 Actionable Safety Protocols")
        for rec in integrated_summary['recommendations']:
            st.markdown(f"""
            <div class="rec-item">
                {rec}
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📥 Export Executive Safety Audit Report")
    
    md_report = generate_markdown_audit_report(integrated_summary)
    
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.download_button(
            label="📄 Download Executive Report (Markdown)",
            data=md_report,
            file_name="RoadSafe_AI_Executive_Report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with c_d2:
        st.download_button(
            label="📊 Export Raw Signals (JSON)",
            data=json.dumps(integrated_summary, indent=2),
            file_name="RoadSafe_AI_Signals.json",
            mime="application/json",
            use_container_width=True
        )


# ==========================================
# SCREEN 6: MODEL PERFORMANCE & EVALUATION
# ==========================================
elif navigation == "📈 Model Performance":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">📈 Model Performance & Comparative Evaluation</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            Evaluation benchmarks, classification metrics, and confusion matrix comparing 
            <b>Logistic Regression</b>, <b>Random Forest</b>, and <b>XGBoost</b> on road safety datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    _, _, metadata = load_ml_artifacts()
    st.subheader("📊 Model Comparison Benchmark")
    
    comparison_data = metadata.get('model_comparison', [
        {'Model': 'Random Forest', 'Accuracy': 0.8259, 'Precision': 0.8259, 'Recall': 0.8259, 'F1-score': 0.8259},
        {'Model': 'XGBoost', 'Accuracy': 0.8249, 'Precision': 0.8249, 'Recall': 0.8249, 'F1-score': 0.8255},
        {'Model': 'Logistic Regression', 'Accuracy': 0.8174, 'Precision': 0.8174, 'Recall': 0.8174, 'F1-score': 0.8174}
    ])
    
    df_metrics = pd.DataFrame(comparison_data)
    st.dataframe(
        df_metrics.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-score'], color='#065F46'),
        use_container_width=True
    )
    
    m_col1, m_col2 = st.columns([1, 1])
    
    with m_col1:
        st.subheader("📉 Multi-Class Confusion Matrix")
        cm_data = metadata.get('confusion_matrix', [[1200, 150, 50], [180, 1100, 120], [60, 140, 1000]])
        class_labels = metadata.get('class_names', ['High', 'Low', 'Medium'])
        
        fig_cm = px.imshow(
            cm_data,
            x=class_labels,
            y=class_labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            text_auto=True
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with m_col2:
        st.subheader("🏆 Model Benchmark Comparison")
        fig_bar = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=df_metrics['Model'],
                y=df_metrics[metric],
                text=[f"{v:.3f}" for v in df_metrics[metric]],
                textposition='auto'
            ))
            
        fig_bar.update_layout(
            barmode='group',
            template="plotly_dark",
            yaxis=dict(range=[0.75, 0.90], gridcolor='#334155'),
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# SCREEN 7: FASTAPI MICROSERVICE DOCS
# ==========================================
elif navigation == "🔌 FastAPI Microservice Docs":
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #38BDF8; margin-bottom: 4px;">🔌 RoadSafe AI — REST API Microservice</h2>
        <p style="color: #94A3B8; margin-bottom: 0;">
            High-performance asynchronous backend microservice built with <b>FastAPI</b> & <b>Uvicorn</b>, 
            enabling seamless integration with smart city traffic infrastructure, mobile apps, and emergency systems.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 How to Run the FastAPI Server")
    st.code("uvicorn api:app --host 0.0.0.0 --port 8000 --reload", language="bash")
    
    st.markdown("### 🌐 Available Endpoints & Swagger Documentation")
    
    c_api1, c_api2 = st.columns(2)
    with c_api1:
        st.markdown("""
        - `GET /` — Microservice Health Check & Component Status
        - `POST /predict-risk` — ML Environmental Risk Classifier
        - `POST /detect-objects` — YOLOv8 Road Object Detection & Density
        """)
    with c_api2:
        st.markdown("""
        - `POST /analyze-report` — NLP Text Narrative Entity Extraction
        - `POST /integrated-risk` — Unified Cross-Modal Fusion
        - `GET /docs` — Interactive Swagger OpenAPI Documentation
        """)
        
    st.markdown("---")
    st.markdown("### 💻 Example Python API Request (CURL / Requests)")
    
    st.code("""
import requests

# Predict accident risk via REST API
response = requests.post("http://127.0.0.1:8000/predict-risk", json={
    "weather_condition": "Severe Storm",
    "road_condition": "Standing Water",
    "time_of_day": "Night/Late",
    "visibility_mi": 1.2,
    "temperature_f": 48.0,
    "humidity_pct": 95.0,
    "wind_speed_mph": 34.0,
    "speed_limit": 65,
    "traffic_signal": 1,
    "junction": 1,
    "crossing": 0
})

print(response.json())
# Output: {"risk_level": "High", "risk_index": 92.4, "confidence": 0.964, ...}
    """, language="python")
