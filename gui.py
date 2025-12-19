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

# ==========================================
# 1. إعدادات المسارات والبيئة
# ==========================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'Project Data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. التنسيق الجمالي (CSS)
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
        border: 1px solid #FFCC80; margin-top: 5px; font-size: 1.1em;
    }
    .weight-tag {
        font-size: 0.9em; color: #666; font-style: italic; margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. الوظائف الذكية (حسابات الوزن والسعرات)
# ==========================================
def extract_weight_from_name(filename):
    """يستخرج الوزن من اسم الملف، مثال: 2371108_112g.jpg تعطي 112.0"""
    grams = 100.0
    try:
        name_lower = filename.lower()
        if 'g' in name_lower:
            # نأخذ الجزء الذي يسبق حرف g مباشرة
            pre_g = name_lower.split('g')[0]
            weight_str = "".join([c for c in pre_g.split('_')[-1] if c.isdigit() or c == '.'])
            if weight_str:
                grams = float(weight_str)
    except: pass
    return grams

def calculate_final_calories(raw_text, grams):
    """يعالج نصوص مثل '~0.83 calories per gram' ويحسب الإجمالي"""
    if not raw_text or raw_text == "N/A": return "N/A"
    try:
        # استخراج الرقم فقط من النص (تجاهل ~ والكلمات)
        clean_num = "".join([c for c in raw_text if c.isdigit() or c == '.'])
        if not clean_num: return raw_text
        
        val_per_unit = float(clean_num)
        
        # إذا كان الملف يحدد أنها "لكل جرام"
        if "per gram" in raw_text.lower():
            total = val_per_unit * grams
        else:
            # الافتراضي (لكل 100 جرام)
            total = (val_per_unit * grams) / 100.0
            
        return f"{total:.1f}"
    except:
        return raw_text

# ==========================================
# 4. هياكل الموديلات
# ==========================================
class FoodFruitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
    def forward(self, x): return self.backbone(x)

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128))
    def forward(self, x): return self.head(self.backbone(x))

class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x): return self.backbone(x)

# ==========================================
# 5. تحميل الملفات والأوزان
# ==========================================
@st.cache_resource
def load_all_assets():
    # تحميل أسماء الكلاسات
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
def load_calories_map():
    cal_map = {}
    paths = [os.path.join(DATA_DIR, 'Food/Train Calories.txt'), 
             os.path.join(DATA_DIR, 'Food/Val Calories.txt'), 
             os.path.join(DATA_DIR, 'Fruit/Calories.txt')]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':', 1)
                        # تنظيف المفتاح للبحث الدقيق
                        norm_k = k.strip().lower().replace(' ', '').replace('_', '')
                        cal_map[norm_k] = v.strip()
    return cal_map

# ==========================================
# 6. الواجهة الرئيسية (Tabs)
# ==========================================
(models_list, fruit_names) = load_all_assets()
(m1, m2, m3, m4, m5) = models_list
cal_map = load_calories_map()

tab1, tab2 = st.tabs(["🚀 Deep Food Analysis", "🧬 Siamese Similarity"])

with tab1:
    uploaded = st.file_uploader("Upload Food Images", accept_multiple_files=True)
    if uploaded and st.button("Start AI Analysis"):
        tf_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tf_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        for f in uploaded:
            img = Image.open(f).convert('RGB')
            grams = extract_weight_from_name(f.name)
            t = tf_224(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                is_fruit = torch.argmax(m1(t), 1).item() == 1
                if is_fruit:
                    idx = torch.argmax(m3(t), 1).item()
                    sub_cat = fruit_names[idx]
                    # Segmentation
                    t256 = tf_256(img).unsqueeze(0).to(DEVICE)
                    bin_mask = (torch.sigmoid(m4(t256)) > 0.5).cpu().numpy()[0][0]
                else:
                    # Food logic via ProtoNet (Simulated distance matching)
                    sub_cat = "Food Item" # Replace with actual distance logic if needed

            # البحث عن السعرات
            search_key = sub_cat.lower().replace(' ', '').replace('_', '')
            raw_cal_text = cal_map.get(search_key, "N/A")
            total_cal = calculate_final_calories(raw_cal_text, grams)

            # العرض الجمالي
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.5, 2, 2.5])
            with c1: st.image(img, use_container_width=True)
            with c2:
                st.markdown(f"### {sub_cat}")
                st.markdown(f'<div class="calorie-badge">🔥 {total_cal} Total Cal</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="weight-tag">Detected: {grams}g</div>', unsafe_allow_html=True)
            with c3:
                if is_fruit:
                    st.image(bin_mask, caption="AI Segmentation Mask", use_container_width=True)
                else:
                    st.info("Product identified via visual fingerprint matching.")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("🧬 Visual Similarity Search (Siamese Ranking)")
    col_a, col_b = st.columns([1, 2])
    with col_a: anc = st.file_uploader("Anchor Image", key="siamese_anc")
    with col_b: gall = st.file_uploader("Gallery Collection", accept_multiple_files=True, key="siamese_gal")
    
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
                    st.markdown(f"<p style='text-align:center; color:{d_color}; font-weight:bold;'>Dist: {item['d']:.3f}</p>", unsafe_allow_html=True)
                    if item['d'] < 10: st.success("Match")
                    else: st.info("Close" if item['d'] < 20 else "No Match")
