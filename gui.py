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
# 1. الإعدادات والتنسيق الجمالي (CSS)
# ==========================================
st.set_page_config(page_title="AI Food Analyzer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .result-card {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 10px solid #FF4B4B;
        margin-bottom: 20px;
    }
    .calorie-badge { 
        background-color: #FFF3E0; color: #E65100; padding: 8px 15px; 
        border-radius: 30px; font-weight: bold; display: inline-block;
        border: 1px solid #FFCC80; margin: 5px;
    }
    .siamese-card {
        border: 1px solid #ddd; border-radius: 12px; padding: 10px; 
        background-color: #ffffff; text-align: center; margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# المسارات
BASE_PATH = "D:/fcis/Cv/Computer-Vision-Project-main"
MODEL_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'Project Data')
OUTPUT_DIR = os.path.join(BASE_PATH, 'Integrated_Test_Results')
PROTO_CACHE_FILE = os.path.join(MODEL_DIR, 'food_prototypes.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. وظائف المساعدة
# ==========================================
def extract_weight(filename):
    try:
        match = re.search(r'(\d+)g', filename.lower())
        if match: return float(match.group(1))
    except: pass
    return 100.0

def get_final_calories(raw_text, grams):
    if not raw_text or raw_text == "N/A": return "N/A"
    try:
        num_part = "".join([c for c in raw_text if c.isdigit() or c == '.'])
        val_per_unit = float(num_part)
        if "per gram" in raw_text.lower():
            total = val_per_unit * grams
        else:
            total = (val_per_unit * grams) / 100.0
        return f"{total:.1f}"
    except: return raw_text

def get_colored_mask(mask_indices, num_classes=31):
    h, w = mask_indices.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    cmap = plt.get_cmap('gist_ncar', num_classes)
    for cls_idx in range(1, num_classes):
        color = np.array(cmap(cls_idx)[:3]) * 255
        color_mask[mask_indices == cls_idx] = color.astype(np.uint8)
    return color_mask

# ==========================================
# 3. هياكل الموديلات وتحميلها
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

@st.cache_resource
def load_all_assets():
    with open(os.path.join(MODEL_DIR, 'part_c_classes.json'), 'r') as f:
        fruit_names = json.load(f)['class_names']
    
    def load_sd(model, name):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            sd = torch.load(p, map_location=DEVICE)
            model.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd, strict=False)
            model.to(DEVICE).eval()
            
    m1 = FoodFruitClassifier(); load_sd(m1, 'part_a_best.pth')
    m2 = ProtoNet(); load_sd(m2, 'protonet_food_model.pth')
    m3 = FruitClassifier(len(fruit_names)); load_sd(m3, 'part_c_best.pth')
    
    # موديل السيجمنتيشن الجديد Unet++ بمقاس 320
    m4 = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    load_sd(m4, 'best_model_unetplusplus.pth')
    
    # موديل المالتي سيجمنتيشن بمقاس 256
    m5 = smp.Unet("resnet34", in_channels=3, classes=31); load_sd(m5, 'best_fruit_segmentation.pth')
    
    return (m1, m2, m3, m4, m5), fruit_names

@st.cache_resource
def load_cal_map():
    cal_map = {}
    files = [os.path.join(DATA_DIR, 'Food/Train Calories.txt'), os.path.join(DATA_DIR, 'Food/Val Calories.txt'), os.path.join(DATA_DIR, 'Fruit/Calories.txt')]
    for fp in files:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':', 1)
                        norm_k = k.strip().lower().replace(' ', '').replace('_', '')
                        cal_map[norm_k] = v.strip()
    return cal_map

# ==========================================
# 4. المعالجة والعرض
# ==========================================
(m1, m2, m3, m4, m5), fruit_classes = load_all_assets()
cal_map = load_cal_map()
food_protos = pickle.load(open(PROTO_CACHE_FILE, 'rb')) if os.path.exists(PROTO_CACHE_FILE) else {}

st.title("🍎 Food AI Intelligence Pro")
tab1, tab2 = st.tabs(["🚀 Comprehensive Analysis", "🧬 Visual Similarity Search"])

with tab1:
    uploaded = st.file_uploader("Upload Images", accept_multiple_files=True, key="tab1_up")
    if uploaded and st.button("🚀 Run AI Analysis"):
        # إعداد الترانسفورمز
        tf_cls = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tf_seg_320 = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()]) # لـ Unet++
        tf_seg_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]) # للمالتي

        for file in uploaded:
            img_name = Path(file.name).stem
            img_folder = os.path.join(OUTPUT_DIR, img_name)
            os.makedirs(img_folder, exist_ok=True)

            img_pil = Image.open(file).convert('RGB')
            grams = extract_weight(file.name)
            img_t_cls = tf_cls(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # تصنيف (طعام/فاكهة)
                is_fruit = torch.argmax(m1(img_t_cls), 1).item() == 1
                classification_label = "Fruit" if is_fruit else "Food"
                
                if is_fruit:
                    sub_cat = fruit_classes[torch.argmax(m3(img_t_cls), 1).item()]
                    
                    # Binary Mask بمقاس 320
                    img_t_320 = tf_seg_320(img_pil).unsqueeze(0).to(DEVICE)
                    b_mask_tensor = torch.sigmoid(m4(img_t_320))
                    b_mask_np = (b_mask_tensor > 0.5).float().cpu().numpy()[0][0]
                    
                    # Multi-class Mask بمقاس 256
                    img_t_256 = tf_seg_256(img_pil).unsqueeze(0).to(DEVICE)
                    m_mask_idx = torch.argmax(m5(img_t_256), 1).cpu().numpy()[0]
                    colored_mask = get_colored_mask(m_mask_idx)

                    plt.imsave(os.path.join(img_folder, "binary_mask.png"), b_mask_np, cmap='gray')
                    Image.fromarray(colored_mask).save(os.path.join(img_folder, "multi_mask.png"))
                else:
                    emb = m2(img_t_cls).cpu()
                    sub_cat, min_d = "Unknown", float('inf')
                    for name, proto in food_protos.items():
                        d = torch.norm(emb - proto).item()
                        if d < min_d: min_d, sub_cat = d, name

            raw_val = cal_map.get(sub_cat.lower().replace(' ', '').replace('_', ''), "N/A")
            total_cal = get_final_calories(raw_val, grams)

            with open(os.path.join(img_folder, f"{img_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"{classification_label}\n{sub_cat}\n{total_cal}\n")

            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.write(f"📁 Folder: `{img_name}/` (Binary: 320px | Multi: 256px)")
            c1, c2, c3 = st.columns([1.5, 2, 3.5])
            c1.image(img_pil, use_container_width=True, caption=file.name)
            with c2:
                st.markdown(f"### {sub_cat} ({classification_label})")
                st.markdown(f'<div class="calorie-badge">⚖️ {grams} g</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="calorie-badge">🔥 {total_cal} Cal</div>', unsafe_allow_html=True)
            with c3:
                if is_fruit:
                    mc1, mc2 = st.columns(2)
                    mc1.image(b_mask_np, caption="Binary (Unet++) 320px", use_container_width=True)
                    mc2.image(colored_mask, caption="Multi-class 256px", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# الجزء الخاص بـ Similarity Search (Tab 2) يبقى كما هو في كودك الأصلي
with tab2:
    st.subheader("🧬 Visual Similarity Search")
    col_a, col_b = st.columns([1, 2])
    with col_a: anc = st.file_uploader("Anchor Image", key="siamese_anc")
    with col_b: gall = st.file_uploader("Gallery Collection", accept_multiple_files=True, key="siamese_gall")
    if anc and gall:
        tf_siam = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            a_emb = m2(tf_siam(Image.open(anc).convert('RGB')).unsqueeze(0).to(DEVICE))
            results = []
            for gf in gall:
                g_emb = m2(tf_siam(Image.open(gf).convert('RGB')).unsqueeze(0).to(DEVICE))
                dist = torch.norm(a_emb - g_emb).item()
                results.append({'file': gf, 'name': gf.name, 'dist': dist})
            results.sort(key=lambda x: x['dist'])
            st.divider()
            cols = st.columns(4)
            for i, item in enumerate(results):
                with cols[i % 4]:
                    st.markdown(f'<div class="siamese-card"><b>{item["name"]}</b></div>', unsafe_allow_html=True)
                    st.image(item['file'], use_container_width=True)
                    d_color = "green" if item['dist'] < 10 else "orange" if item['dist'] < 20 else "red"
                    st.markdown(f"<p style='text-align:center; color:{d_color};'><b>Dist: {item['dist']:.3f}</b></p>", unsafe_allow_html=True)