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
# 1. إعدادات الصفحة والتنسيق الجمالي (CSS)
# ==========================================
st.set_page_config(page_title="AI Food Analyzer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    
    /* تصميم بطاقة النتيجة */
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 10px solid #FF4B4B;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* تنسيق السعرات */
    .calorie-badge { 
        background-color: #FFF3E0; 
        color: #E65100; 
        padding: 8px 15px; 
        border-radius: 30px; 
        font-weight: bold;
        font-size: 1.1em;
        display: inline-block;
        border: 1px solid #FFCC80;
    }
    
    .stHeader { color: #1f1f1f; }
    </style>
    """, unsafe_allow_html=True)

# المسارات
BASE_PATH = "D:/fcis/Cv/Computer-Vision-Project-main"
MODEL_DIR = os.path.join(BASE_PATH, 'models')
OUTPUT_DIR = os.path.join(BASE_PATH, 'Integrated_Test_Results')
PROTO_CACHE_FILE = os.path.join(MODEL_DIR, 'food_prototypes.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. هياكل الموديلات والوظائف المساعدة
# ==========================================
# [تعريف الكلاسات FoodFruitClassifier, ProtoNet, FruitClassifier]
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
    CALORIE_FILES = [os.path.join(BASE_PATH, 'Project Data/Food/Train Calories.txt'), os.path.join(BASE_PATH, 'Project Data/Food/Val Calories.txt'), os.path.join(BASE_PATH, 'Project Data/Fruit/Calories.txt')]
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
    FOOD_DIRS = [os.path.join(BASE_PATH, 'Project Data/Food/Train'), os.path.join(BASE_PATH, 'Project Data/Food/Validation')]
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
    with open(PROTO_CACHE_FILE, 'wb') as f: pickle.dump(prototypes, f)
    return prototypes

# ==========================================
# 3. واجهة المستخدم (The GUI)
# ==========================================
st.title("🍎 Food AI Intelligence Pro")
st.markdown("CLASSIFICATION | SEGMENTATION | CALORIES | SIMILARITY")

tab1, tab2 = st.tabs(["🚀 Comprehensive Analysis", "🧬 Visual Similarity Search"])

with tab1:
    st.subheader("Step 1: Upload Images for Analysis")
    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, key="main_upload")
    
    if uploaded_files:
        if st.button("🚀 Start AI Deep Analysis"):
            (m1, m2, m3, m4, m5), fruit_names = load_assets()
            food_protos = get_cached_prototypes(m2)
            cal_map = load_all_calories_map()
            
            tf_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            tf_256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            st.markdown("### Processing Results")
            prog = st.progress(0)
            
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
                    
                    with open(os.path.join(current_img_dir, f"{img_stem}.txt"), 'w', encoding='utf-8') as f:
                        f.write(f"{main_lbl}\n{sub_cat}\n{cal_val}")

                # --- بطاقة عرض النتيجة ---
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1.5, 2, 3])
                
                with c1:
                    st.image(img_pil, use_container_width=True)
                    st.caption(f"File: {file.name}")
                
                with c2:
                    color = "#2E7D32" if is_fruit else "#1565C0"
                    st.markdown(f"<h2 style='color:{color}; margin-top:0;'>{main_lbl}</h2>", unsafe_allow_html=True)
                    st.markdown(f"**Identified as:** {sub_cat}")
                    st.markdown(f'<div class="calorie-badge">🔥 {cal_val} Cal</div>', unsafe_allow_html=True)
                    st.write(f"📁 Exported to: `/{img_stem}`")
                
                with c3:
                    if is_fruit:
                        m_c1, m_c2 = st.columns(2)
                        m_c1.image(os.path.join(current_img_dir, "binary_mask.png"), caption="Binary Segment")
                        m_c2.image(os.path.join(current_img_dir, "multi_mask.png"), caption="Multi-Class Color")
                    else:
                        st.info("🧬 Feature Matching Logic: Object recognized via visual fingerprinting against dataset prototypes.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                prog.progress((idx + 1) / len(uploaded_files))
            st.balloons()

with tab2:
    st.subheader("Visual Similarity Comparison")
    st.write("Find and rank similar objects using Siamese distance.")
    
    col_anc, col_gal = st.columns([1, 2])
    with col_anc:
        anc = st.file_uploader("Anchor Image", key="s_anc_main")
    with col_gal:
        gall = st.file_uploader("Gallery Images", accept_multiple_files=True, key="s_gal_main")
        
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
                    st.image(item['f'], use_container_width=True)
                    st.markdown(f"**{item['n']}**")
                    st.metric("Similarity", f"{item['d']:.3f}")
                    if item['d'] < 10: st.success("Match Found")
                    else: st.error("No Match")