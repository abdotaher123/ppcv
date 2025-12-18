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
# 1. إعدادات المسارات الديناميكية (مهم جداً للرفع)
# ==========================================
# السطر القادم يحدد مسار الفولدر الحالي تلقائياً أياً كان مكانه
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'Project Data')
OUTPUT_DIR = os.path.join(BASE_PATH, 'Integrated_Test_Results')
PROTO_CACHE_FILE = os.path.join(MODEL_DIR, 'food_prototypes.pkl')

# التأكد من وجود مجلد النتائج
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. التنسيق الجمالي (CSS)
# ==========================================
st.set_page_config(page_title="AI Food Analyzer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 10px solid #FF4B4B;
        margin-top: 20px; margin-bottom: 20px;
    }
    .calorie-badge { 
        background-color: #FFF3E0; color: #E65100; 
        padding: 8px 15px; border-radius: 30px; 
        font-weight: bold; font-size: 1.1em;
        display: inline-block; border: 1px solid #FFCC80;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. هياكل الموديلات
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
# 4. وظائف التحميل والمعالجة
# ==========================================
def get_colored_mask(mask_indices, num_classes=31):
    h, w = mask_indices.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    cmap = plt.get_cmap('gist_ncar', num_classes)
    for cls_idx in range(1, num_classes):
        color = np.array(cmap(cls_idx)[:3]) * 255
        color_mask[mask_indices == cls_idx] = color.astype(np.uint8)
    return color_mask

@st.cache_resource
def load_assets():
    # تحميل أسماء الكلاسات
    json_path = os.path.join(MODEL_DIR, 'part_c_classes.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            fruit_classes = json.load(f)['class_names']
    else:
        fruit_classes = [f"Fruit_{i}" for i in range(30)] # Fallback
        
    def load_sd(model, name):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            sd = torch.load(p, map_location=DEVICE)
            model.load_state_dict(sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd, strict=False)
            model.to(DEVICE).eval()
        else:
            st.error(f"Missing model file: {name}")
    
    m1 = FoodFruitClassifier(); load_sd(m1, 'part_a_best.pth')
    m2 = ProtoNet(); load_sd(m2, 'protonet_food_model.pth')
    # لاحظ هنا تم تغيير الاسم لـ part_c_fixed بناء على صورتك لتقليل الحجم
    m3 = FruitClassifier(len(fruit_classes)); load_sd(m3, 'part_c_fixed.pth') 
    m4 = smp.Unet("resnet34", in_channels=3, classes=1); load_sd(m4, 'best_model.pth')
    m5 = smp.Unet("resnet34", in_channels=3, classes=31); load_sd(m5, 'best_fruit_segmentation.pth')
    return (m1, m2, m3, m4, m5), fruit_classes

@st.cache_resource
def load_all_calories_map():
    cal_map = {}
    CALORIE_FILES = [
        os.path.join(DATA_DIR, 'Food/Train Calories.txt'),
        os.path.join(DATA_DIR, 'Food/Val Calories.txt'),
        os.path.join(DATA_DIR, 'Fruit/Calories.txt')
    ]
    for fp in CALORIE_FILES:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        k, v = line.strip().split(':', 1)
                        normalized_key = k.strip().lower().replace(' ', '').replace('_', '')
                        cal_map[normalized_key] = v.strip()
    return cal_map

@st.cache_resource
def get_cached_prototypes(_m2):
    if os.path.exists(PROTO_CACHE_FILE):
        with open(PROTO_CACHE_FILE, 'rb') as f: return pickle.load(f)
    prototypes = {}
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    FOOD_DIRS = [os.path.join(DATA_DIR, 'Food/Train'), os.path.join(DATA_DIR, 'Food/Validation')]
    for folder in FOOD_DIRS:
        if os.path.exists(folder):
            for cls in os.listdir(folder):
                cls_p = os.path.join(folder, cls)
                if os.path.isdir(cls_p):
                    imgs = [f for f in os.listdir(cls_p) if f.lower().endswith(('.jpg', '.png'))][:5]
                    embs = []
                    for n in imgs:
                        img_t = tf(Image.open(os.path.join(cls_p, n)).convert('RGB')).unsqueeze(0).to(DEVICE)
                        with torch.no_grad(): embs.append(_m2(img_t).cpu())
                    if embs: prototypes[cls] = torch.mean(torch.stack(embs), dim=0)
    if prototypes:
        with open(PROTO_CACHE_FILE, 'wb') as f: pickle.dump(prototypes, f)
    return prototypes

# ==========================================
# 5. الواجهة (The GUI)
# ==========================================
st.title("🍎 Food AI Intelligence Pro")
tab1, tab2 = st.tabs(["🚀 Comprehensive Analysis", "🧬 Visual Similarity Search"])

with tab1:
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Execute Analysis"):
        (m1, m2, m3, m4, m5), fruit_names = load_assets()
        food_protos = get_cached_prototypes(m2)
        cal_map = load_all_calories_map()
        
        tf_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tf_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        for idx, file in enumerate(uploaded_files):
            img_stem = Path(file.name).stem
            current_img_dir = os.path.join(OUTPUT_DIR, img_stem)
            os.makedirs(current_img_dir, exist_ok=True)
            
            img_pil = Image.open(file).convert('RGB')
            img_t = tf_224(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                is_fruit = torch.argmax(m1(img_t), 1).item() == 1
                main_lbl = "Fruit" if is_fruit else "Food"
                
                if is_fruit:
                    sub_cat = fruit_names[torch.argmax(m3(img_t), 1).item()]
                    img_256 = tf_256(img_pil).unsqueeze(0).to(DEVICE)
                    # Masks
                    b_mask = (torch.sigmoid(m4(img_256)) > 0.5).float().cpu().numpy()[0][0]
                    plt.imsave(os.path.join(current_img_dir, "binary_mask.png"), b_mask, cmap='gray')
                    m_mask_indices = torch.argmax(m5(img_256), 1).cpu().numpy()[0]
                    colored_mask = get_colored_mask(m_mask_indices)
                    Image.fromarray(colored_mask).save(os.path.join(current_img_dir, "multi_mask.png"))
                else:
                    emb = m2(img_t).cpu()
                    sub_cat, min_d = "Unknown", float('inf')
                    for name, proto in food_protos.items():
                        dist = torch.norm(emb - proto).item()
                        if dist < min_d: min_d, sub_cat = dist, name

                search_key = sub_cat.lower().replace(' ', '').replace('_', '')
                cal_val = cal_map.get(search_key, "N/A")

            # عرض النتائج في بطاقة
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.5, 2, 3])
            with c1: st.image(img_pil, use_container_width=True)
            with c2:
                st.markdown(f"## {main_lbl}")
                st.write(f"**Identified:** {sub_cat}")
                st.markdown(f'<div class="calorie-badge">🔥 {cal_val} Cal</div>', unsafe_allow_html=True)
            with c3:
                if is_fruit:
                    mc1, mc2 = st.columns(2)
                    mc1.image(os.path.join(current_img_dir, "binary_mask.png"), caption="Binary")
                    mc2.image(os.path.join(current_img_dir, "multi_mask.png"), caption="Multi-Class")
            st.markdown('</div>', unsafe_allow_html=True)
        st.balloons()

with tab2:
    st.subheader("Visual Similarity Search")
    anc = st.file_uploader("Anchor Image")
    gall = st.file_uploader("Gallery Collection", accept_multiple_files=True)
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
            cols = st.columns(4)
            for i, item in enumerate(results):
                with cols[i % 4]:
                    st.image(item['f'], use_container_width=True)
                    st.metric("Dist", f"{item['d']:.2f}")
