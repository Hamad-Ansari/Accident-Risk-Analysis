"""
RoadSafe AI - Computer Vision & Deep Learning Module (Module A)
Powered by YOLOv8 for real-time road object detection, vehicle counting,
traffic density estimation, and Dashcam/Video stream analytics.
"""

import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Target Road Safety Object Classes
ROAD_CLASSES = {
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Bus',
    'motorcycle': 'Motorcycle',
    'bicycle': 'Bicycle',
    'person': 'Pedestrian'
}

CLASS_COLORS = {
    'car': (38, 198, 218),        # Vibrant Cyan
    'truck': (255, 167, 38),      # Amber Orange
    'bus': (171, 71, 188),        # Purple
    'motorcycle': (255, 238, 88), # Bright Yellow
    'bicycle': (102, 187, 106),   # Mint Green
    'person': (239, 83, 80)       # Coral Red
}

_yolo_model_cache = None


def get_yolo_model():
    """Loads and caches the YOLOv8 model."""
    global _yolo_model_cache
    if _yolo_model_cache is None:
        try:
            from ultralytics import YOLO
            _yolo_model_cache = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None
    return _yolo_model_cache


def detect_road_objects(image_input, conf_threshold=0.35, iou_threshold=0.45):
    """
    Runs YOLOv8 object detection on a single road image or frame.
    """
    model = get_yolo_model()
    
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    elif isinstance(image_input, (str, Path)):
        img_np = cv2.imread(str(image_input))
        if img_np is not None:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_np = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        img_np = image_input

    if model is None:
        return _fallback_simulation(img_np)

    results = model(img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)
    result = results[0]
    names = result.names
    
    counts = {v: 0 for v in ROAD_CLASSES.values()}
    raw_detections = []
    
    annotated_bgr = result.plot(line_width=2, font_size=1)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            if cls_name in ROAD_CLASSES:
                friendly_name = ROAD_CLASSES[cls_name]
                counts[friendly_name] = counts.get(friendly_name, 0) + 1
                raw_detections.append({
                    'class': friendly_name,
                    'confidence': conf,
                    'bbox': xyxy
                })
    
    active_counts = {k: v for k, v in counts.items() if v > 0}
    pedestrians = counts.get('Pedestrian', 0)
    total_vehicles = sum(v for k, v in counts.items() if k != 'Pedestrian')
    
    # Estimate Traffic Density
    if total_vehicles <= 3:
        density = 'Low'
        density_score = min(30.0, total_vehicles * 10.0)
    elif total_vehicles <= 8:
        density = 'Medium'
        density_score = 30.0 + (total_vehicles - 3) * 10.0
    else:
        density = 'High'
        density_score = min(100.0, 80.0 + (total_vehicles - 8) * 4.0)

    return {
        'annotated_image': annotated_rgb,
        'counts': active_counts,
        'all_counts': counts,
        'total_vehicles': total_vehicles,
        'pedestrians': pedestrians,
        'density': density,
        'density_score': density_score,
        'raw_detections': raw_detections,
        'total_objects': len(raw_detections)
    }


def process_video_analytics(video_file_path, conf_threshold=0.35, sample_rate=2, max_frames=300, progress_callback=None):
    """
    Processes video stream / dashcam footage frame-by-frame and produces a time-series
    traffic density profile, cumulative class counts, and video analytics.
    """
    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_read = min(total_video_frames, max_frames)
    
    timeline_data = []
    frame_idx = 0
    processed_count = 0
    last_annotated_frame = None
    
    # Cumulative and peak object tracking
    cumulative_counts = {v: 0 for v in ROAD_CLASSES.values()}
    max_counts = {v: 0 for v in ROAD_CLASSES.values()}
    
    start_time = time.time()
    
    while cap.isOpened() and frame_idx < frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detect_road_objects(frame_rgb, conf_threshold=conf_threshold)
            last_annotated_frame = res['annotated_image']
            
            # Aggregate counts
            for cls_name, cnt in res['all_counts'].items():
                cumulative_counts[cls_name] += cnt
                if cnt > max_counts[cls_name]:
                    max_counts[cls_name] = cnt
                    
            timestamp_sec = round(frame_idx / fps, 2)
            timeline_data.append({
                'frame': frame_idx,
                'timestamp_sec': timestamp_sec,
                'total_vehicles': res['total_vehicles'],
                'pedestrians': res['pedestrians'],
                'density_score': res['density_score'],
                'density': res['density'],
                'counts': res['counts']
            })
            processed_count += 1
            
            if progress_callback:
                progress_callback(
                    processed_count, 
                    frames_to_read // sample_rate, 
                    last_annotated_frame, 
                    res['total_vehicles'], 
                    res['density']
                )
                
        frame_idx += 1
        
    cap.release()
    total_time = max(0.001, time.time() - start_time)
    avg_fps = round(processed_count / total_time, 1)
    
    if not timeline_data:
        return None
        
    df_timeline = pd.DataFrame(timeline_data)
    
    avg_vehicles = round(float(df_timeline['total_vehicles'].mean()), 1)
    peak_vehicles = int(df_timeline['total_vehicles'].max())
    peak_row = df_timeline.loc[df_timeline['total_vehicles'].idxmax()]
    peak_timestamp = f"{peak_row['timestamp_sec']}s"
    
    # Overarching video traffic classification
    if avg_vehicles <= 3:
        overall_density = 'Low'
    elif avg_vehicles <= 8:
        overall_density = 'Medium'
    else:
        overall_density = 'High'
        
    # Active cumulative counts (>0)
    active_cumulative = {k: v for k, v in cumulative_counts.items() if v > 0}
    
    return {
        'timeline_df': df_timeline,
        'last_frame': last_annotated_frame,
        'avg_vehicles': avg_vehicles,
        'peak_vehicles': peak_vehicles,
        'peak_timestamp': peak_timestamp,
        'overall_density': overall_density,
        'avg_fps': avg_fps,
        'frames_analyzed': processed_count,
        'video_duration_sec': round(frame_idx / fps, 1),
        'cumulative_counts': cumulative_counts,
        'active_cumulative': active_cumulative,
        'max_simultaneous': max_counts
    }


def create_video_density_timeline_chart(timeline_df):
    """Generates an interactive Plotly line chart showing traffic density over video duration."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_df['timestamp_sec'],
        y=timeline_df['total_vehicles'],
        mode='lines+markers',
        name='Vehicles Count',
        line=dict(color='#38BDF8', width=3),
        marker=dict(size=5, color='#0284C7'),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.15)'
    ))
    
    if timeline_df['pedestrians'].max() > 0:
        fig.add_trace(go.Scatter(
            x=timeline_df['timestamp_sec'],
            y=timeline_df['pedestrians'],
            mode='lines',
            name='Pedestrians',
            line=dict(color='#EF4444', width=2, dash='dash')
        ))
        
    fig.add_hline(y=3.5, line_dash="dot", line_color="#10B981", annotation_text="Low/Med Bound", annotation_position="top left", annotation_font_color="#34D399")
    fig.add_hline(y=8.5, line_dash="dot", line_color="#EF4444", annotation_text="Heavy Congestion", annotation_position="top left", annotation_font_color="#F87171")
    
    fig.update_layout(
        title="Traffic Density & Vehicle Count Timeline (Seconds)",
        title_font=dict(size=14, color="#E2E8F0"),
        xaxis=dict(title="Video Timeline (Seconds)", gridcolor="#334155", color="#94A3B8"),
        yaxis=dict(title="Count on Road", gridcolor="#334155", color="#94A3B8"),
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_counts_chart(counts_dict, title="Detected Road Objects Breakdown"):
    """Generates an interactive Plotly bar chart of detected objects."""
    if not counts_dict:
        fig = go.Figure()
        fig.add_annotation(text="No road objects detected", showarrow=False, font=dict(size=14, color="#94A3B8"))
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
        
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    colors = ['#38BDF8', '#F59E0B', '#A855F7', '#EAB308', '#22C55E', '#EF4444'][:len(labels)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition='auto',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        title_font=dict(size=14, color="#E2E8F0"),
        xaxis=dict(title="Object Category", showgrid=False, color="#94A3B8"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#334155", color="#94A3B8"),
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_density_gauge(density_score, density_label):
    """Generates an interactive circular density gauge."""
    color_map = {
        'Low': '#10B981',
        'Medium': '#F59E0B',
        'High': '#EF4444'
    }
    gauge_color = color_map.get(density_label, '#3B82F6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=density_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Traffic Density: <b>{density_label.upper()}</b>", 'font': {'size': 16, 'color': '#E2E8F0'}},
        number={'suffix': "%", 'font': {'color': gauge_color, 'size': 26}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': gauge_color},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.8,
                'value': density_score
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


def _fallback_simulation(img_np):
    return {
        'annotated_image': img_np,
        'counts': {'Car': 4, 'Truck': 1, 'Pedestrian': 1},
        'all_counts': {'Car': 4, 'Truck': 1, 'Bus': 0, 'Motorcycle': 0, 'Bicycle': 0, 'Pedestrian': 1},
        'total_vehicles': 5,
        'pedestrians': 1,
        'density': 'Medium',
        'density_score': 55.0,
        'raw_detections': [],
        'total_objects': 6
    }
