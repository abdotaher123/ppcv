import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import pickle
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. إعدادات المسارات والبيئة
# ==========================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'Project Data')
OUTPUT_DIR = "/tmp/Integrated_Test_Results"
PROTO_CACHE_FILE = os.path.join(MODEL_DIR, 'food_prototypes.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. التنسيق الجمالي الشامل (CSS)
# ==========================================
st.set_page_config(page_title="AI Food Analyzer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .result-card {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 10px solid #FF4B4B;
        margin-bottom: 25px;
    }
    .siamese-card {
        border: 1px solid #ddd; border-radius: 12px; padding: 10px; 
        background-color: #ffffff; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        text-align: center; margin-bottom: 10px;
    }
    .file-name-text {
        font-weight: bold; font-size: 0.85em; color: #444;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .calorie-badge { 
        background-color: #FFF3E0; color: #E65100; padding: 8px 15px; 
        border-radius: 30px; font-weight: bold; display: inline-block;
        border: 1px solid #FFCC80; margin-top: 5px;
    }
    .weight-tag {
        font-size: 0.9em; color: #666; font-style: italic; margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. الوظائف الذكية (استخراج الوزن والسعرات)
# ==========================================
def extract_weight_from_name(filename):
    """يستخرج الوزن من اسم الملف، مثال: 2371108_112g.jpg تعطي 112.0"""
    grams = 100.0
    try:
        match = re.search(r'(\d+)g', filename.lower())
        if match:
            grams = float(match.group(1))
    except: pass
    return grams

def clean_calorie_value(raw_val, grams):
    """يعالج نصوص السعرات مثل '~0.83 calories per gram'"""
    if not raw_val or raw_val == "N/A": return "N/A"
    try:
        clean_num = "".join([c for c in raw_val if c.isdigit() or c == '.'])
        if not clean_num: return raw_val
        base_val = float(clean_num)
        
        # الحساب بناءً على وحدة 'per gram' الموجودة في ملفك
        if "per gram" in raw_val.lower():
            total = base_val * grams
        else:
            total = (base_val * grams) / 100.0
        return f"{total:.1f}"
    except:
        return raw_val

# ==========================================
# 4. هياكل الموديلات
# ==========================================
class FoodFruitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.fc.in_features, num_classes))
    def forward(self, x): return self.backbone(x)

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128))
    def forward(self, x): return self.head(self.backbone(x))

class FruitClassifier(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    def forward(self, x): return self.backbone(x)

# ==========================================
# 5. تحميل الأصول والبيانات
# ==========================================
@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, 'part_c_classes.json'), 'r') as f:
        fruit_classes = json.load(f)['class_names']
    
    def load_sd(model, name):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            sd = torch.load(p, map_location=DEVICE)
            model.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd, strict=False)
            model.to(DEVICE).eval()
    
    m1 = FoodFruitClassifier(); load_sd(m1, 'part_a_best.pth')
    m2 = ProtoNet(); load_sd(m2, 'protonet_food_model.pth')
    m3 = FruitClassifier(len(fruit_classes)); load_sd(m3, 'part_c_best.pth')
    m4 = smp.Unet("resnet34", in_channels=3, classes=1); load_sd(m4, 'best_model.pth')
    m5 = smp.Unet("resnet34", in_channels=3, classes=31); load_sd(m5, 'best_fruit_segmentation.pth')
    return (m1, m2, m3, m4, m5), fruit_classes

@st.cache_resource
def load_all_calories_map():
    cal_map = {}
    paths = [os.path.join(DATA_DIR, 'Food/Train Calories.txt'), os.path.join(DATA_DIR, 'Food/Val Calories.txt'), os.path.join(DATA_DIR, 'Fruit/Calories.txt')]
    for fp in paths:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':', 1)
                        norm_k = k.strip().lower().replace(' ', '').replace('_', '')
                        cal_map[norm_k] = v.strip()
    return cal_map

# ==========================================
# 6. الواجهة البرمجية (Tabs)
# ==========================================
st.title("🍎 Food AI Intelligence Pro")
tab1, tab2 = st.tabs(["🚀 Analysis & Segmentation", "🧬 Visual Similarity Search"])

with tab1:
    uploaded_files = st.file_uploader("Upload Food Images", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Run Deep Analysis"):
        (m1, m2, m3, m4, m5), fruit_names = load_assets()
        cal_map = load_all_calories_map()
        food_protos = pickle.load(open(PROTO_CACHE_FILE, 'rb')) if os.path.exists(PROTO_CACHE_FILE) else {}
        
        tf_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tf_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        for file in uploaded_files:
            img_pil = Image.open(file).convert('RGB')
            grams = extract_weight_from_name(file.name)
            img_t = tf_224(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                is_fruit = torch.argmax(m1(img_t), 1).item() == 1
                if is_fruit:
                    sub_cat = fruit_names[torch.argmax(m3(img_t), 1).item()]
                    t256 = tf_256(img_pil).unsqueeze(0).to(DEVICE)
                    b_mask = (torch.sigmoid(m4(t256)) > 0.5).cpu().numpy()[0][0]
                else:
                    emb = m2(img_t).cpu()
                    sub_cat, min_d = "Unknown", float('inf')
                    for name, proto in food_protos.items():
                        d = torch.norm(emb - proto).item()
                        if d < min_d: min_d, sub_cat = d, name

            raw_cal = cal_map.get(sub_cat.lower().replace(' ', '').replace('_', ''), "N/A")
            final_cal = clean_calorie_value(raw_cal, grams)

            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.5, 2, 3])
            c1.image(img_pil, use_container_width=True)
            with c2:
                st.markdown(f"### {sub_cat}")
                st.markdown(f'<div class="calorie-badge">🔥 {final_cal} Total Cal</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="weight-tag">Detected Weight: {grams}g</div>', unsafe_allow_html=True)
            with c3:
                if is_fruit: st.image(b_mask, caption="Segmentation Mask", use_container_width=True)
                else: st.info("Identified via prototype matching.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.balloons()

with tab2:
    st.subheader("🧬 Visual Similarity Search (Siamese Ranking)")
    col_a, col_b = st.columns([1, 2])
    with col_a: anc = st.file_uploader("Anchor Image (Target)", key="siamese_anc")
    with col_b: gall = st.file_uploader("Gallery (Collection to search)", accept_multiple_files=True, key="siamese_gal")
    
    if anc and gall:
        (m1, m2, m3, m4, m5), _ = load_assets()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            a_emb = m2(tf(Image.open(anc).convert('RGB')).unsqueeze(0).to(DEVICE))
            results = []
            for gf in gall:
                g_emb = m2(tf(Image.open(gf).convert('RGB')).unsqueeze(0).to(DEVICE))
                dist = torch.norm(a_emb - g_emb).item()
                results.append({'f': gf, 'n': gf.name, 'd': dist})
            
            results.sort(key=lambda x: x['d'])
            st.divider()
            cols = st.columns(4)
            for i, item in enumerate(results):
                with cols[i % 4]:
                    st.markdown(f'<div class="siamese-card"><div class="file-name-text">📄 {item["n"]}</div></div>', unsafe_allow_html=True)
                    st.image(item['f'], use_container_width=True)
                    d_color = "green" if item['d'] < 10 else "orange" if item['d'] < 20 else "red"
                    st.markdown(f"<p style='text-align:center; color:{d_color}; font-weight:bold;'>Distance: {item['d']:.3f}</p>", unsafe_allow_html=True)
                    if item['d'] < 10: st.success("Strong Match")
                    elif item['d'] < 20: st.warning("Similar Appearance")
                    else: st.error("No Match")
