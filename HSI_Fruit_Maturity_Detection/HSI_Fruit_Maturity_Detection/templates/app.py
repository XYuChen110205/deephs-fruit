import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import tempfile
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='.')

# -------------------------- 加载训练集归一化参数 --------------------------
DATA_NPZ = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'avocado_detection', 'vis_2d_dataset_v2.npz')
if os.path.exists(DATA_NPZ):
    _data_bundle = np.load(DATA_NPZ, allow_pickle=True)
    VIS_MEAN = _data_bundle['train_mean']
    VIS_STD  = _data_bundle['train_std']
    print(f"[INFO] Normalization params loaded from {DATA_NPZ}")
else:
    print(f"[WARN] {DATA_NPZ} not found, using hardcoded fallback")
    VIS_MEAN = np.zeros((224, 1, 1), dtype=np.float32)
    VIS_STD  = np.ones((224, 1, 1), dtype=np.float32)


# -------------------------- 模型定义（与训练时一致）--------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
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
            nn.Conv2d(bands, bands * k, 7, padding=3, groups=bands),
            nn.Conv2d(bands * k, hidden_layers[0], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[0]),
            SEBlock(hidden_layers[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers[0], hidden_layers[0]*k, 3, padding=1, groups=hidden_layers[0]),
            nn.Conv2d(hidden_layers[0]*k, hidden_layers[1], 1),
            nn.ReLU(True), nn.AvgPool2d(2),
            nn.BatchNorm2d(hidden_layers[1]),
            SEBlock(hidden_layers[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_layers[1], hidden_layers[1]*k, 3, padding=1, groups=hidden_layers[1]),
            nn.Conv2d(hidden_layers[1]*k, hidden_layers[2], 1),
            nn.ReLU(True), nn.BatchNorm2d(hidden_layers[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(hidden_layers[2], num_classes)
        )

    def forward(self, x):
        return self.fc(self.gap(self.block3(self.block2(self.block1(x)))).flatten(1))


# -------------------------- 全局加载模型 --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassifierNetwork(bands=224, num_classes=3).to(device)
le = LabelEncoder()
le.classes_ = np.array(['overripe', 'perfect', 'unripe'])  # 字母序，与训练时一致

MODEL_PATH = '../avocado_detection/avocado_hs_cnn_v2_ema.pth'
model_loaded = False
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model_loaded = True
    print(f"[INFO] Model loaded from {MODEL_PATH}")
else:
    print(f"[WARN] {MODEL_PATH} not found. /predict will return error.")


# -------------------------- 波长列表（Specim FX10 VIS 相机）--------------------------
WAVELENGTHS = np.arange(397.66, 397.66 + 224 * 2.68, 2.68)[:224]

# -------------------------- 预处理函数 --------------------------
def preprocess(hdr_path, bin_path):
    import spectral
    img = spectral.envi.open(hdr_path, image=bin_path).load()
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, -1e10, 1e10)
    if img.ndim < 3:
        raise ValueError("图像维度不足")

    n_bands = img.shape[-1]
    if n_bands != 224:
        raise ValueError(
            f"检测到 {n_bands} 个光谱波段，但模型需要 224 波段（VIS 相机）。"
            f"请上传 VIS 相机数据（day_01~day_10 中的 .hdr/.bin 文件），"
            f"不要使用 NIR 相机数据。正确路径示例："
            f" data/avocado/VIS/day_01/"
        )

    t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(62, 62), mode='bilinear', align_corners=False)
    arr = t.squeeze(0).numpy()

    padded = np.zeros((arr.shape[0], 64, 64), dtype=np.float32)
    padded[:, 1:-1, 1:-1] = arr
    padded = (padded - VIS_MEAN) / VIS_STD
    padded = np.clip(np.nan_to_num(padded), -10, 10)
    return padded, arr


# -------------------------- 推断函数（TTA × 5）--------------------------
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
    return pred_label, mean_prob, le.classes_


# -------------------------- 高光谱可视化：光谱曲线 + 假彩色 PCA 合成 --------------------------
def compute_spectral_curve(raw_data):
    """计算全帧平均光谱曲线"""
    try:
        h, w = raw_data.shape[1], raw_data.shape[2]
        center_h, center_w = h // 2, w // 2
        crop_h = min(15, h // 4)
        crop_w = min(15, w // 4)
        crop = raw_data[:, center_h-crop_h:center_h+crop_h, center_w-crop_w:center_w+crop_w]
        spectrum = crop.mean(axis=(1, 2))
        return spectrum.tolist()
    except Exception as e:
        print(f"[WARN] Spectral curve failed: {e}")
        return None


def compute_false_color_pca(raw_data):
    """用 PCA 将 224 波段映射到 3 通道假彩色图像，并放大到 256x256"""
    try:
        from sklearn.decomposition import PCA
        h, w = raw_data.shape[1], raw_data.shape[2]
        pixels = raw_data.reshape(raw_data.shape[0], -1).T
        pca = PCA(n_components=3)
        components = pca.fit_transform(pixels)
        comp_min, comp_max = components.min(axis=0), components.max(axis=0)
        comp_range = comp_max - comp_min
        comp_range[comp_range == 0] = 1
        normalized = ((components - comp_min) / comp_range * 255).astype(np.uint8)
        image = normalized.reshape(h, w, 3)
        import base64, io
        from PIL import Image
        pil_img = Image.fromarray(image, 'RGB')
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[WARN] PCA false color failed: {e}")
        return None


# -------------------------- Grad-CAM 热力图 --------------------------
def compute_gradcam(x_np, target_class=None):
    """生成 Grad-CAM 热力图，叠加在 RGB 图像上"""
    x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
    x.requires_grad = True

    feature_maps = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal feature_maps
        feature_maps = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_f = model.block3.register_forward_hook(forward_hook)
    handle_b = model.block3.register_full_backward_hook(backward_hook)

    out = model(x)
    if target_class is None:
        target_class = out.argmax(1).item()
    model.zero_grad()
    out[0, target_class].backward()

    handle_f.remove()
    handle_b.remove()

    if feature_maps is None or gradients is None:
        return None

    weights = gradients.squeeze(0).mean(dim=(1, 2))
    cam = torch.zeros(feature_maps.shape[2:], device=device)
    for i, w in enumerate(weights):
        cam += w * feature_maps.squeeze(0)[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_np = cam.detach().cpu().numpy()
    cam_np = np.clip(cam_np, 0, 1)

    # 放大到 256x256
    import cv2
    cam_resized = cv2.resize(cam_np, (256, 256), interpolation=cv2.INTER_LINEAR)
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 生成 RGB 背景
    r_idx = int(np.argmin(np.abs(WAVELENGTHS - 660)))
    g_idx = int(np.argmin(np.abs(WAVELENGTHS - 550)))
    b_idx = int(np.argmin(np.abs(WAVELENGTHS - 470)))

    raw_raw = x_np  # 已经归一化
    rgb = np.stack([raw_raw[r_idx], raw_raw[g_idx], raw_raw[b_idx]], axis=-1)
    for c in range(3):
        mn, mx = rgb[:,:,c].min(), rgb[:,:,c].max()
        if mx > mn:
            rgb[:,:,c] = (rgb[:,:,c] - mn) / (mx - mn) * 255
    rgb = rgb.astype(np.uint8)
    rgb_resized = cv2.resize(rgb, (256, 256))

    # 叠加
    overlay = cv2.addWeighted(rgb_resized, 0.5, cam_colored_rgb, 0.5, 0)

    import base64, io
    from PIL import Image as PILImg
    pil_img = PILImg.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# -------------------------- 路由 --------------------------
@app.route('/')
def index():
    return render_template('index_v2.html', model_loaded=model_loaded)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': f'模型文件 {MODEL_PATH} 未找到，请先训练模型'}), 500

    if 'hdr' not in request.files or 'bin' not in request.files:
        return jsonify({'error': '请同时上传 .hdr 和 .bin 文件'}), 400

    hdr_file = request.files['hdr']
    bin_file = request.files['bin']

    with tempfile.TemporaryDirectory() as tmpdir:
        hdr_path = os.path.join(tmpdir, 'upload.hdr')
        bin_path = os.path.join(tmpdir, 'upload.bin')
        hdr_file.save(hdr_path)
        bin_file.save(bin_path)

        try:
            x, raw = preprocess(hdr_path, bin_path)
        except Exception as e:
            return jsonify({'error': f'预处理失败：{str(e)}'}), 400

        label, probs, classes = predict_tta(x)
        spectrum = compute_spectral_curve(raw) if raw is not None else None
        false_color = compute_false_color_pca(raw) if raw is not None else None
        gradcam_heatmap = compute_gradcam(x) if raw is not None else None

    label_map = {'unripe': '未成熟', 'perfect': '完熟', 'overripe': '过熟'}
    result = {
        'label': label,
        'label_cn': label_map.get(label, label),
        'confidence': float(probs.max()),
        'probabilities': {
            cls: float(p) for cls, p in zip(classes, probs)
        },
        'spectral_curve': spectrum,
        'false_color_image': false_color,
        'gradcam_heatmap': gradcam_heatmap,
        'wavelengths': WAVELENGTHS.tolist()
    }
    return jsonify(result)


@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model_loaded,
        'device': str(device),
        'model_path': MODEL_PATH
    })


@app.route('/api/ablation')
def ablation_data():
    json_path = os.path.join(os.path.dirname(__file__), 'ablation_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Ablation results not found'}), 404


@app.route('/api/comparison')
def comparison_data():
    json_path = os.path.join(os.path.dirname(__file__), 'comparison_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Comparison results not found'}), 404


@app.route('/api/records', methods=['GET'])
def get_records():
    records_path = os.path.join(os.path.dirname(__file__), 'detection_records.json')
    if os.path.exists(records_path):
        with open(records_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route('/api/records', methods=['POST'])
def save_records():
    records_path = os.path.join(os.path.dirname(__file__), 'detection_records.json')
    records = request.get_json()
    with open(records_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)