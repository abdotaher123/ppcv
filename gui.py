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
# 1. إعدادات المسارات الديناميكية (مهم جداً للعمل على السيرفر)
# ==========================================
# الحصول على المسار الحالي للمجلد الذي يحتوي على الكود
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_PATH, 'models')
DATA_DIR = os.path.join(BASE_PATH, 'Project Data')
# استخدام مجلد tmp للحفظ لأنه المجلد الوحيد المسموح بالكتابة فيه في Hugging Face
OUTPUT_DIR = "/tmp/Integrated_Test_Results"
PROTO_CACHE_FILE = os.path.join(MODEL_DIR, 'food_prototypes.pkl')

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
    json_path = os.path.join(MODEL_DIR, 'part_c_classes.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            fruit_classes = json.load(f)['class_names']
    else:
        fruit_classes = [f"Class_{i}" for i in range(30)]
        
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
    m3 = FruitClassifier(len(fruit_classes)); load_sd(m3, 'part_c_best.pth')
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
    return {} # في حال عدم وجود صور للتدريب يدوياً، يفضل رفع ملف .pkl جاهز

# ==========================================
# 5. الواجهة (The GUI)
# ==========================================
st.title("🍎 Food AI Intelligence Pro")
tab1, tab2 = st.tabs(["🚀 Comprehensive Analysis", "🧬 Visual Similarity Search"])

with tab1:
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Start AI Deep Analysis"):
        (m1, m2, m3, m4, m5), fruit_names = load_assets()
        food_protos = get_cached_prototypes(m2)
        cal_map = load_all_calories_map()
        
        tf_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tf_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        for idx, file in enumerate(uploaded_files):
            img_stem = Path(file.name).stem
            img_pil = Image.open(file).convert('RGB')
            img_t = tf_224(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                is_fruit = torch.argmax(m1(img_t), 1).item() == 1
                main_lbl = "Fruit" if is_fruit else "Food"
                
                if is_fruit:
                    sub_cat = fruit_names[torch.argmax(m3(img_t), 1).item()]
                    img_256 = tf_256(img_pil).unsqueeze(0).to(DEVICE)
                    b_mask = (torch.sigmoid(m4(img_256)) > 0.5).float().cpu().numpy()[0][0]
                    m_mask_indices = torch.argmax(m5(img_256), 1).cpu().numpy()[0]
                    colored_mask = get_colored_mask(m_mask_indices)
                else:
                    emb = m2(img_t).cpu()
                    sub_cat, min_d = "Unknown", float('inf')
                    for name, proto in food_protos.items():
                        dist = torch.norm(emb - proto).item()
                        if dist < min_d: min_d, sub_cat = dist, name

                search_key = sub_cat.lower().replace(' ', '').replace('_', '')
                cal_val = cal_map.get(search_key, "N/A")

            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.5, 2, 3])
            with c1: st.image(img_pil, use_container_width=True)
            with c2:
                color = "#2E7D32" if is_fruit else "#1565C0"
                st.markdown(f"<h2 style='color:{color};'>{main_lbl}</h2>", unsafe_allow_html=True)
                st.write(f"**Identified:** {sub_cat}")
                st.markdown(f'<div class="calorie-badge">🔥 {cal_val} Cal</div>', unsafe_allow_html=True)
            with c3:
                if is_fruit:
                    mc1, mc2 = st.columns(2)
                    mc1.image(b_mask, caption="Binary", use_container_width=True)
                    mc2.image(colored_mask, caption="Multi-Class", use_container_width=True)
                else:
                    st.info("Visual fingerprinting used for identification.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.balloons()
with tab2:
    st.subheader("🧬 Visual Similarity Search")
    st.write("Find and rank similar objects using Siamese distance embeddings.")
    
    col_anc, col_gal = st.columns([1, 2])
    with col_anc:
        anc = st.file_uploader("📸 Anchor Image", key="s_anc_main")
    with col_gal:
        gall = st.file_uploader("🖼️ Gallery Collection", accept_multiple_files=True, key="s_gal_main")
        
    if anc and gall:
        (m1, m2, m3, m4, m5), _ = load_assets()
        tf = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            # استخراج بصمة الصورة المرجعية
            a_emb = m2(tf(Image.open(anc).convert('RGB')).unsqueeze(0).to(DEVICE))
            results = []
            
            # معالجة جاليري الصور
            for gf in gall:
                g_emb = m2(tf(Image.open(gf).convert('RGB')).unsqueeze(0).to(DEVICE))
                dist = torch.norm(a_emb - g_emb).item()
                results.append({'f': gf, 'n': gf.name, 'd': dist})
            
            # ترتيب النتائج من الأقرب للأبعد
            results.sort(key=lambda x: x['d'])
            
            st.divider()
            st.markdown("### 📊 Similarity Ranking (Most Similar First)")
            
            # عرض النتائج في شبكة (Grid)
            cols = st.columns(4)
            for i, item in enumerate(results):
                with cols[i % 4]:
                    # تصميم كارت صغير لكل نتيجة
                    st.markdown(f"""
                        <div style="
                            border: 1px solid #ddd; 
                            border-radius: 10px; 
                            padding: 10px; 
                            background-color: white; 
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
                            margin-bottom: 15px;
                            text-align: center;
                        ">
                            <p style="margin: 0; font-weight: bold; font-size: 0.85em; color: #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                📄 {item['n']}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(item['f'], use_container_width=True)
                    
                    # عرض المسافة بلون متغير حسب الدقة
                    dist_color = "green" if item['d'] < 10 else "orange" if item['d'] < 20 else "red"
                    st.markdown(f"<p style='text-align:center; color:{dist_color}; font-weight:bold;'>Distance: {item['d']:.3f}</p>", unsafe_allow_html=True)
                    
                    if item['d'] < 10:
                        st.success("✅ Match")
                    elif item['d'] < 20:
                        st.warning("⚠️ Likely")
                    else:
                        st.error("❌ Different")}")
