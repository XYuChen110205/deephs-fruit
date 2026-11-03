"""
Streamlit 部署版 — HS-CNN 高光谱成熟度检测系统
替代原来的 Flask + HTML 方案，单文件即可运行
"""
import os
import json
import base64
import io
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from PIL import Image

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ========== 页面配置 ==========
st.set_page_config(
    page_title="HS-CNN 高光谱成熟度检测",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 模型定义（与训练时一致）==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.gap(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ClassifierNetwork(nn.Module):
    def __init__(self, bands, num_classes=3, hidden_layers=[25, 30, 50]):
        super().__init__()
        k = 3
        self.block1 = nn.Sequential(
            nn.Conv2d(bands, bands*k, 7, padding=3, groups=bands),
            nn.Conv2d(bands*k, hidden_layers[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[0]), SEBlock(hidden_layers[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers[0], hidden_layers[0]*k, 3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0]*k, hidden_layers[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[1]), SEBlock(hidden_layers[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_layers[1], hidden_layers[1]*k, 3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1]*k, hidden_layers[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden_layers[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden_layers[2], num_classes)
        )
    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))


# ========== 模型加载（缓存）==========
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierNetwork(bands=224, num_classes=3).to(device)
    le = LabelEncoder()
    le.classes_ = np.array(['overripe', 'perfect', 'unripe'])

    # 找模型文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "avocado_hs_cnn_v2_ema.pth")
    npz_path = os.path.join(base_dir, "vis_2d_dataset_v2.npz")

    if not os.path.exists(model_path):
        # 尝试从上级目录找
        model_path = os.path.join(base_dir, "..", "avocado_detection", "avocado_hs_cnn_v2_ema.pth")
        npz_path = os.path.join(base_dir, "..", "avocado_detection", "vis_2d_dataset_v2.npz")

    model_loaded = False
    vis_mean, vis_std = None, None
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        vis_mean = data['train_mean']
        vis_std = data['train_std']
    return model, le, device, model_loaded, vis_mean, vis_std

model, le, device, model_loaded, vis_mean, vis_std = load_model()

# ========== 波长列表 ==========
WAVELENGTHS = np.arange(397.66, 397.66 + 224 * 2.68, 2.68)[:224]
LABEL_MAP = {'unripe': '未成熟', 'perfect': '完熟', 'overripe': '过熟'}
CLASS_ORDER = ['unripe', 'perfect', 'overripe']
CLASS_COLORS = {'unripe': '#66BB6A', 'perfect': '#FFA726', 'overripe': '#EF5350'}


# ========== 预处理 ==========
def preprocess(hdr_path, bin_path):
    import spectral
    img = spectral.envi.open(hdr_path, image=bin_path).load()
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, -1e10, 1e10)
    if img.ndim < 3:
        raise ValueError("图像维度不足")
    n_bands = img.shape[-1]
    if n_bands != 224:
        raise ValueError(f"检测到 {n_bands} 波段，需要 224 波段（VIS 相机）")
    t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False)
    arr = t.squeeze(0).numpy()
    padded = np.zeros((arr.shape[0], 64, 64), dtype=np.float32)
    padded[:, 1:-1, 1:-1] = arr
    if vis_mean is not None:
        padded = (padded - vis_mean) / vis_std
    padded = np.clip(np.nan_to_num(padded), -10, 10)
    return padded, arr


def predict_tta(x_np, tta_iter=5):
    x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
    all_probs = []
    with torch.no_grad():
        for i in range(tta_iter):
            xi = x.clone()
            if i > 0:
                if torch.rand(1) > 0.5: xi = torch.flip(xi, [2])
                if torch.rand(1) > 0.5: xi = torch.flip(xi, [3])
                if torch.rand(1) > 0.5: xi = torch.rot90(xi, 1, [2, 3])
            prob = F.softmax(model(xi), dim=1).cpu().numpy()[0]
            all_probs.append(prob)
    mean_prob = np.mean(all_probs, axis=0)
    pred_idx = mean_prob.argmax()
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label, mean_prob


def compute_spectral_curve(raw_data):
    h, w = raw_data.shape[1], raw_data.shape[2]
    ch, cw = h // 2, w // 2
    crop = raw_data[:, ch-15:ch+15, cw-15:cw+15]
    return crop.mean(axis=(1, 2))


def compute_false_color_pca(raw_data):
    h, w = raw_data.shape[1], raw_data.shape[2]
    pixels = raw_data.reshape(raw_data.shape[0], -1).T
    pca = PCA(n_components=3)
    components = pca.fit_transform(pixels)
    comp_min, comp_max = components.min(axis=0), components.max(axis=0)
    cr = comp_max - comp_min
    cr[cr == 0] = 1
    normalized = ((components - comp_min) / cr * 255).astype(np.uint8)
    img = normalized.reshape(h, w, 3)
    pil_img = Image.fromarray(img, 'RGB').resize((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return pil_img


def compute_rgb(raw_data):
    """从224波段中提取R(660nm)、G(550nm)、B(470nm)合成真彩色"""
    r_idx = np.argmin(np.abs(WAVELENGTHS - 660))
    g_idx = np.argmin(np.abs(WAVELENGTHS - 550))
    b_idx = np.argmin(np.abs(WAVELENGTHS - 470))
    rgb = np.stack([raw_data[r_idx], raw_data[g_idx], raw_data[b_idx]], axis=-1)
    # 归一化到 0-255
    for c in range(3):
        mn, mx = rgb[:,:,c].min(), rgb[:,:,c].max()
        if mx > mn:
            rgb[:,:,c] = (rgb[:,:,c] - mn) / (mx - mn) * 255
    rgb = rgb.astype(np.uint8)
    pil_img = Image.fromarray(rgb, 'RGB').resize((256, 256), Image.LANCZOS)
    return pil_img


# ========== 消融实验/对比实验数据 ==========
def load_ablation_data():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "ablation_results.json")
    if not os.path.exists(p):
        p = os.path.join(base, "..", "templates", "ablation_results.json")
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_comparison_data():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "comparison_results.json")
    if not os.path.exists(p):
        p = os.path.join(base, "..", "templates", "comparison_results.json")
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ==========================================
#  Streamlit 页面
# ==========================================
st.sidebar.markdown("## 🥑 HS-CNN 高光谱检测")
st.sidebar.markdown("---")
page = st.sidebar.radio("导航", ["成熟度检测", "消融实验", "对比实验"])

# ---- 状态指示 ----
status_color = "🟢" if model_loaded else "🔴"
status_text = "模型就绪" if model_loaded else "模型未加载"
st.sidebar.markdown(f"**状态**: {status_color} {status_text}")
st.sidebar.markdown(f"**设备**: {device}")
st.sidebar.markdown(f"**模型**: HS-CNN + SE Attention v2")

# ========== 页面1: 成熟度检测 ==========
if page == "成熟度检测":
    st.title("🥑 牛油果成熟度检测")
    st.markdown("上传 Specim FX10 高光谱相机的 `.hdr` + `.bin` 文件对进行无损检测")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 📁 文件上传")
        hdr_file = st.file_uploader("选择 .hdr 头文件", type=['hdr'])
        bin_file = st.file_uploader("选择 .bin 数据文件", type=['bin', 'raw'])

        if hdr_file and bin_file:
            st.success(f"已选择: {hdr_file.name} + {bin_file.name}")
            if st.button("🚀 开始检测", type="primary", use_container_width=True):
                with st.spinner("模型推理中..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            hdr_path = os.path.join(tmpdir, 'upload.hdr')
                            bin_path = os.path.join(tmpdir, 'upload.bin')
                            with open(hdr_path, 'wb') as f: f.write(hdr_file.getbuffer())
                            with open(bin_path, 'wb') as f: f.write(bin_file.getbuffer())

                            x, raw = preprocess(hdr_path, bin_path)
                            label, probs = predict_tta(x)

                        # 结果显示
                        label_cn = LABEL_MAP.get(label, label)
                        confidence = float(probs.max())

                        # 光谱曲线
                        spectrum = compute_spectral_curve(raw)
                        # 假彩色
                        pca_img = compute_false_color_pca(raw)
                        # 真彩色
                        rgb_img = compute_rgb(raw)

                        st.session_state['result'] = {
                            'label': label,
                            'label_cn': label_cn,
                            'confidence': confidence,
                            'probs': {c: float(p) for c, p in zip(CLASS_ORDER, probs)},
                            'spectrum': spectrum,
                            'pca_img': pca_img,
                            'rgb_img': rgb_img,
                        }
                    except Exception as e:
                        st.error(f"检测失败: {e}")

        elif hdr_file and not bin_file:
            st.warning("请同时上传 .bin 文件")
        elif bin_file and not hdr_file:
            st.warning("请同时上传 .hdr 文件")
        else:
            st.info("请选择 VIS 相机拍摄的牛油果高光谱数据")

    with col2:
        if 'result' in st.session_state:
            r = st.session_state['result']

            # 成熟度 + 置信度
            st.markdown("### 📊 检测结果")
            conf_color = "green" if r['confidence'] > 0.8 else "orange" if r['confidence'] > 0.6 else "red"
            st.markdown(
                f"<div style='padding:20px;background:#F1F8E9;border-radius:10px;border:1px solid #E0E0E0'>"
                f"<h2 style='margin:0;color:#2E7D32'>{r['label_cn']}</h2>"
                f"<p style='color:#8D6E63;font-size:14px'>牛油果 · {r['label'].upper()}</p>"
                f"<p style='font-size:28px;font-weight:700;color:{conf_color}'>{r['confidence']*100:.1f}%</p>"
                f"<p style='font-size:12px;color:#8D6E63'>置信度</p>"
                f"</div>", unsafe_allow_html=True
            )

            # 概率条
            st.markdown("#### 概率分布")
            for cls in CLASS_ORDER:
                p = r['probs'][cls]
                st.progress(p, text=f"**{LABEL_MAP[cls]}** {p*100:.1f}%")

            # 光谱曲线
            st.markdown("#### 🌈 全波段光谱曲线 (224 bands)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.round(WAVELENGTHS).tolist(),
                y=r['spectrum'].tolist(),
                mode='lines',
                name='平均反射率',
                line=dict(color='#2E7D32', width=1.5),
                hovertemplate='波长: %{x} nm<br>反射率: %{y:.4f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="波长 (nm)",
                yaxis_title="归一化反射率",
                height=250, margin=dict(l=0, r=0, t=0, b=0),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # 图像对比
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 🎨 真彩色 (RGB)")
                st.image(r['rgb_img'], use_container_width=True)
                st.caption("从224波段中提取 R(660nm)/G(550nm)/B(470nm) 合成")
            with col_b:
                st.markdown("#### 🔬 假彩色 (PCA)")
                st.image(r['pca_img'], use_container_width=True)
                st.caption("PCA将224波段降维到3通道，保留最大信息量")


# ========== 页面2: 消融实验 ==========
elif page == "消融实验":
    st.title("🧪 消融实验 (Ablation Study)")
    data = load_ablation_data()
    if data and data.get('ablation_results'):
        ds = data['dataset']
        results = data['ablation_results']

        st.markdown("### 数据集概况")
        cols = st.columns(5)
        cols[0].metric("训练样本", ds.get('total_samples', '—'))
        cols[1].metric("光谱波段", ds.get('n_bands', '—'))
        cols[2].metric("类别数", ds.get('n_classes', '—'))
        cols[3].metric("验证样本", ds.get('val_samples', 15))
        cols[4].metric("测试样本", ds.get('test_samples', 15))

        # 图表
        st.markdown("### 准确率对比")
        names = [r['name'] for r in results]
        sample_acc = [r['sample_acc']*100 for r in results]
        fruit_acc = [r['fruit_acc']*100 for r in results]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='样本准确率', x=names, y=sample_acc,
                             marker_color=['#558B2F','#7CB342','#AED581','#33691E'],
                             text=[f'{v:.1f}%' for v in sample_acc], textposition='outside'))
        fig.add_trace(go.Bar(name='水果准确率', x=names, y=fruit_acc,
                             marker_color=['#33691E','#2E7D32','#388E3C','#1B5E20'],
                             text=[f'{v:.1f}%' for v in fruit_acc], textposition='outside'))
        fig.update_layout(barmode='group', height=400, yaxis_range=[0, 100],
                          yaxis_title="准确率 (%)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # 表格
        df = pd.DataFrame(results)
        df['sample_acc'] = df['sample_acc'].apply(lambda x: f"{x*100:.1f}%")
        df['fruit_acc'] = df['fruit_acc'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(df[['name','sample_acc','fruit_acc','improvement']],
                     column_config={'name': '配置', 'sample_acc': '样本准确率',
                                    'fruit_acc': '水果准确率', 'improvement': '提升'},
                     hide_index=True, use_container_width=True)
    else:
        st.warning("消融实验数据未找到，请确认 ablation_results.json 文件存在")

# ========== 页面3: 对比实验 ==========
else:
    st.title("📊 对比实验 (Comparative Study)")
    data = load_comparison_data()
    if data and data.get('comparison_results'):
        results = data['comparison_results']

        st.markdown("### 数据集概况")
        cols = st.columns(3)
        cols[0].metric("训练样本", data['dataset']['train_samples'])
        cols[1].metric("测试样本", data['dataset']['test_samples'])
        cols[2].metric("方法数量", len(results))

        st.markdown("### 准确率与 F1-Score 对比")
        names = [r['name'] for r in results]
        acc = [r['sample_acc']*100 for r in results]
        f1 = [r['f1_score']*100 for r in results]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='准确率', x=names, y=acc,
                             marker_color=[r['color'] for r in results],
                             text=[f'{v:.1f}%' for v in acc], textposition='outside'))
        fig.add_trace(go.Bar(name='F1-Score', x=names, y=f1,
                             marker_color=['#558B2F']*len(results),
                             text=[f'{v:.1f}%' for v in f1], textposition='outside'))
        fig.update_layout(barmode='group', height=400, yaxis_range=[0, 100],
                          yaxis_title="分数 (%)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # 表格
        df = pd.DataFrame(results)
        for c in ['sample_acc', 'fruit_acc', 'f1_score']:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(df[['name','category','sample_acc','f1_score','description']],
                     column_config={'name': '方法', 'category': '类别',
                                    'sample_acc': '准确率', 'f1_score': 'F1-Score',
                                    'description': '说明'},
                     hide_index=True, use_container_width=True)
    else:
        st.warning("对比实验数据未找到，请确认 comparison_results.json 文件存在")
