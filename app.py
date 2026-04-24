"""
Tuberculosis Detection - Premium Streamlit Dashboard (PyTorch)
"""
import streamlit as st
import os, json, numpy as np, pandas as pd
import plotly.graph_objects as go
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = 128
CLASS_NAMES = ['Normal', 'TB']
DEVICE = torch.device('cpu')

st.set_page_config(page_title="TB Detection AI", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0a0a1a 0%, #0e1117 50%, #1a0a2e 100%); }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0e1117 50%, #1a0a2e 100%); }
.hero-card {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(108,92,231,0.1));
    border: 1px solid rgba(0,212,170,0.3); border-radius: 20px;
    padding: 40px; text-align: center; margin-bottom: 30px; backdrop-filter: blur(10px);
}
.hero-card h1 { font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #6c5ce7, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
.hero-card p { color: #a0a0b0; font-size: 1.1rem; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 24px; text-align: center;
    transition: transform 0.3s, border-color 0.3s;
}
.metric-card:hover { transform: translateY(-4px); border-color: rgba(0,212,170,0.5); }
.metric-value { font-size: 2rem; font-weight: 800; color: #00d4aa; }
.metric-label { font-size: 0.9rem; color: #888; margin-top: 5px; }
.section-header {
    font-size: 1.5rem; font-weight: 700; color: #fff;
    border-left: 4px solid #00d4aa; padding-left: 15px; margin: 30px 0 20px 0;
}
.pred-result {
    background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.3);
    border-radius: 16px; padding: 30px; text-align: center; margin: 20px 0;
}
.pred-result.tb { background: rgba(255,107,107,0.08); border-color: rgba(255,107,107,0.3); }
</style>
""", unsafe_allow_html=True)


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name):
    if name == 'Custom_CNN': return CustomCNN()
    if name == 'ResNet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5), nn.Linear(256, 2))
        return m
    if name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=None)
        inf = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(inf, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3), nn.Linear(256, 2))
        return m


@st.cache_resource
def load_model_cached(name):
    best = os.path.join(MODELS_DIR, f'{name}_best.pth')
    path = os.path.join(MODELS_DIR, f'{name}.pth')
    p = best if os.path.exists(best) else path
    if not os.path.exists(p): return None
    model = build_model(name)
    model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

def load_img(fname):
    p = os.path.join(RESULTS_DIR, fname)
    return Image.open(p) if os.path.exists(p) else None

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Sidebar ---
st.sidebar.markdown("## TB Detection AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "EDA", "Models", "Predict", "Compare"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align:center;color:#666;font-size:0.8rem;'><p>PyTorch Deep Learning</p><p>ResNet50 | EfficientNet | Custom CNN</p></div>", unsafe_allow_html=True)

MODEL_NAMES = ['Custom_CNN', 'ResNet50', 'EfficientNetB0']

if page == "Home":
    st.markdown("<div class='hero-card'><h1>Tuberculosis Detection Using Deep Learning</h1><p>AI-powered chest X-ray analysis for early TB detection using state-of-the-art deep learning</p></div>", unsafe_allow_html=True)

    info = load_json(os.path.join(DATA_DIR, 'preprocessing_info.json'))
    if info:
        cols = st.columns(4)
        for col, (lbl, val, icon) in zip(cols, [("Total Images", info['total_images'], "Images"),
                ("TB Cases", info['tb_count'], "TB"), ("Normal Cases", info['normal_count'], "Normal"), ("Models Trained", 3, "Models")]):
            col.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div><div class='metric-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("""
    **Problem:** Classify chest X-rays as Normal or TB using deep learning.

    **Approach:**
    - Data preprocessing with augmentation
    - Transfer learning (ResNet50, EfficientNetB0)
    - Custom CNN baseline
    - Gradient-based visualization for interpretability
    """)
    c2.markdown("""
    **Technical Stack:**
    - Python & PyTorch
    - Computer Vision (torchvision)
    - Transfer Learning (ImageNet pretrained)
    - Model Evaluation (ROC-AUC, F1, Precision, Recall)
    - Streamlit Dashboard
    """)

    evl = load_json(os.path.join(RESULTS_DIR, 'evaluation_summary.json'))
    if evl:
        st.markdown("<div class='section-header'>Best Model Performance</div>", unsafe_allow_html=True)
        best = max(evl.items(), key=lambda x: x[1].get('test_accuracy', 0))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", best[0])
        c2.metric("Accuracy", f"{best[1]['test_accuracy']*100:.2f}%")
        c3.metric("AUC", f"{best[1].get('auc',0)*100:.2f}%")
        c4.metric("F1-Score", f"{best[1]['weighted_f1']*100:.2f}%")


elif page == "EDA":
    st.markdown("<div class='hero-card'><h1>Exploratory Data Analysis</h1><p>Dataset insights and visualizations</p></div>", unsafe_allow_html=True)
    info = load_json(os.path.join(DATA_DIR, 'preprocessing_info.json'))
    if info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Images", info['total_images'])
        c2.metric("TB Images", f"{info['tb_count']} ({info['tb_count']/info['total_images']*100:.1f}%)")
        c3.metric("Normal Images", f"{info['normal_count']} ({info['normal_count']/info['total_images']*100:.1f}%)")
    for title, fname in [("Class Distribution", "class_distribution.png"), ("Sample X-Ray Images", "sample_images.png"),
            ("Image Dimensions", "image_dimensions.png"), ("Pixel Intensity", "pixel_intensity.png"),
            ("Mean Images per Class", "mean_images.png"), ("Train/Val/Test Split", "split_distribution.png")]:
        img = load_img(fname)
        if img:
            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)


elif page == "Models":
    st.markdown("<div class='hero-card'><h1>Model Training Results</h1><p>Training curves, confusion matrices, and Grad-CAM</p></div>", unsafe_allow_html=True)
    selected = st.selectbox("Select Model", MODEL_NAMES)
    t1, t2, t3, t4 = st.tabs(["Training Curves", "Confusion Matrix", "ROC Curve", "Grad-CAM"])
    for tab, suffix in zip([t1, t2, t3, t4], ['training_curves', 'confusion_matrix', 'roc_curve', 'gradcam']):
        with tab:
            img = load_img(f'{selected}_{suffix}.png')
            if img: st.image(img, use_container_width=True)
            else: st.info("Not available yet.")
    hist = load_json(os.path.join(RESULTS_DIR, f'{selected}_history.json'))
    if hist:
        st.markdown("<div class='section-header'>Training Details</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epochs", len(hist['loss']))
        c2.metric("Best Train Acc", f"{max(hist['accuracy'])*100:.2f}%")
        c3.metric("Best Val Acc", f"{max(hist['val_accuracy'])*100:.2f}%")
        c4.metric("Min Val Loss", f"{min(hist['val_loss']):.4f}")


elif page == "Predict":
    st.markdown("<div class='hero-card'><h1>TB Prediction</h1><p>Upload a chest X-ray for AI analysis</p></div>", unsafe_allow_html=True)
    available = [n for n in MODEL_NAMES
                 if os.path.exists(os.path.join(MODELS_DIR, f'{n}.pth')) or os.path.exists(os.path.join(MODELS_DIR, f'{n}_best.pth'))]
    if not available:
        st.warning("No trained models found. Run model_training.py first.")
    else:
        sel = st.selectbox("Select Model", available, index=available.index('EfficientNetB0') if 'EfficientNetB0' in available else 0)
        uploaded = st.file_uploader("Upload Chest X-Ray Image", type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
        if uploaded:
            c1, c2 = st.columns(2)
            pil_img = Image.open(uploaded).convert('RGB')
            with c1:
                st.markdown("<div class='section-header'>Uploaded Image</div>", unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
            with c2:
                model = load_model_cached(sel)
                if model:
                    inp = preprocess(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        out = model(inp)
                        probs = torch.softmax(out, dim=1)[0]
                    pred_idx = probs.argmax().item()
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = probs[pred_idx].item()
                    css = "tb" if pred_label == "TB" else ""
                    color = "#ff6b6b" if pred_label == "TB" else "#00d4aa"

                    st.markdown(f"""<div class='pred-result {css}'>
                        <h2 style='color:{color}; font-size:2.5rem;'>{pred_label}</h2>
                        <p style='font-size:1.2rem;color:#ccc;'>Confidence: <b style='color:{color};'>{confidence*100:.1f}%</b></p>
                        <p style='color:#888;font-size:0.9rem;'>Model: {sel}</p>
                    </div>""", unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=[probs[0].item()], y=['Normal'], orientation='h', marker_color='#00d4aa', text=[f'{probs[0]*100:.1f}%'], textposition='inside'))
                    fig.add_trace(go.Bar(x=[probs[1].item()], y=['TB'], orientation='h', marker_color='#ff6b6b', text=[f'{probs[1]*100:.1f}%'], textposition='inside'))
                    fig.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)',
                                       plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False,
                                       xaxis=dict(range=[0,1], showgrid=False), yaxis=dict(showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)
                    st.warning("**Disclaimer:** This is an AI tool for educational purposes only. Always consult a medical professional for diagnosis.")


elif page == "Compare":
    st.markdown("<div class='hero-card'><h1>Model Comparison</h1><p>Side-by-side performance analysis</p></div>", unsafe_allow_html=True)
    evl = load_json(os.path.join(RESULTS_DIR, 'evaluation_summary.json'))
    if not evl:
        st.warning("No evaluation results. Run evaluate.py first.")
    else:
        st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
        rows = [{'Model': n, 'Accuracy (%)': round(m['test_accuracy']*100,2), 'AUC (%)': round(m.get('auc',0)*100,2),
                 'F1-Score (%)': round(m['weighted_f1']*100,2), 'Precision (TB)': round(m['precision_tb']*100,2),
                 'Recall (TB)': round(m['recall_tb']*100,2)} for n, m in evl.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df.style.highlight_max(subset=['Accuracy (%)', 'AUC (%)', 'F1-Score (%)'], color='rgba(0,212,170,0.3)'),
                     use_container_width=True, hide_index=True)

        fig = go.Figure()
        for metric, color in [('Accuracy (%)', '#00d4aa'), ('AUC (%)', '#6c5ce7'), ('F1-Score (%)', '#ff6b6b')]:
            fig.add_trace(go.Bar(name=metric, x=df['Model'], y=df[metric], marker_color=color))
        fig.update_layout(barmode='group', height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='white'), legend=dict(orientation='h', y=1.1), yaxis_title='Score (%)',
                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Radar Chart</div>", unsafe_allow_html=True)
        cats = ['Accuracy', 'AUC', 'F1-Score', 'Precision (TB)', 'Recall (TB)']
        fig = go.Figure()
        for (n, m), c in zip(evl.items(), ['#00d4aa', '#ff6b6b', '#ffd93d']):
            vals = [m['test_accuracy']*100, m.get('auc',0)*100, m['weighted_f1']*100, m['precision_tb']*100, m['recall_tb']*100]
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name=n, line=dict(color=c)))
        fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0,100], gridcolor='rgba(255,255,255,0.1)'),
                                      angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')),
                           height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

        for fname in ['all_roc_curves.png', 'model_comparison.png']:
            img = load_img(fname)
            if img:
                st.markdown(f"<div class='section-header'>{'ROC Comparison' if 'roc' in fname else 'Performance Chart'}</div>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
