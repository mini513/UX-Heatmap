import io
import os
import sys
import base64
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.ndimage import gaussian_filter

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
if os.path.isdir(VENDOR_DIR):
    sys.path.insert(0, VENDOR_DIR)

app = Flask(__name__)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None


def load_model():
    global model
    if model is not None:
        return model
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
    model.eval()
    return model


COLORMAPS = {
    'jet': [
        (0.0, (0, 0, 128)), (0.1, (0, 0, 255)), (0.25, (0, 128, 255)),
        (0.4, (0, 255, 255)), (0.5, (128, 255, 128)), (0.6, (255, 255, 0)),
        (0.75, (255, 128, 0)), (0.9, (255, 0, 0)), (1.0, (128, 0, 0)),
    ],
    'hot': [
        (0.0, (0, 0, 0)), (0.33, (200, 0, 0)), (0.66, (255, 180, 0)),
        (1.0, (255, 255, 255)),
    ],
    'inferno': [
        (0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
        (0.75, (249, 142, 9)), (1.0, (252, 255, 164)),
    ],
    'viridis': [
        (0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.0, (253, 231, 37)),
    ],
    'turbo': [
        (0.0, (48, 18, 59)), (0.15, (67, 95, 229)), (0.3, (29, 185, 199)),
        (0.45, (77, 233, 89)), (0.6, (208, 231, 28)), (0.75, (255, 162, 14)),
        (0.9, (222, 60, 10)), (1.0, (122, 4, 3)),
    ],
}


def apply_colormap(data, colormap_name='jet'):
    stops = COLORMAPS.get(colormap_name, COLORMAPS['jet'])
    h, w = data.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = data[y, x]
            for i in range(len(stops) - 1):
                if stops[i][0] <= v <= stops[i + 1][0]:
                    t = (v - stops[i][0]) / (stops[i + 1][0] - stops[i][0] + 1e-8)
                    r = int(stops[i][1][0] * (1 - t) + stops[i + 1][1][0] * t)
                    g = int(stops[i][1][1] * (1 - t) + stops[i + 1][1][1] * t)
                    b = int(stops[i][1][2] * (1 - t) + stops[i + 1][1][2] * t)
                    alpha = int(v * 255)
                    result[y, x] = [r, g, b, alpha]
                    break

    return result


def apply_colormap_vectorized(data, colormap_name='jet'):
    stops = COLORMAPS.get(colormap_name, COLORMAPS['jet'])
    positions = np.array([s[0] for s in stops])
    colors = np.array([s[1] for s in stops], dtype=np.float32)

    h, w = data.shape
    flat = data.flatten()

    indices = np.searchsorted(positions, flat, side='right') - 1
    indices = np.clip(indices, 0, len(positions) - 2)

    t = (flat - positions[indices]) / (positions[indices + 1] - positions[indices] + 1e-8)
    t = np.clip(t, 0, 1)

    c0 = colors[indices]
    c1 = colors[indices + 1]
    rgb = (c0 * (1 - t[:, None]) + c1 * t[:, None]).astype(np.uint8)

    alpha = (flat * 255).astype(np.uint8)

    result = np.zeros((h * w, 4), dtype=np.uint8)
    result[:, :3] = rgb
    result[:, 3] = alpha

    return result.reshape(h, w, 4)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'DeepGaze IIE', 'device': str(DEVICE)})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    colormap = request.form.get('colormap', 'jet')
    blur_sigma = int(request.form.get('blur', 25))

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    mdl = load_model()

    orig_w, orig_h = img.size
    max_dim = 1024
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    centerbias = torch.zeros(1, 1, img_tensor.shape[2], img_tensor.shape[3]).to(DEVICE)

    with torch.no_grad():
        log_density = mdl(img_tensor, centerbias)
        density = torch.exp(log_density)

    heatmap = density.squeeze().cpu().numpy()
    heatmap = gaussian_filter(heatmap, sigma=blur_sigma)

    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    colored = apply_colormap_vectorized(heatmap_resized, colormap)
    heatmap_img = Image.fromarray(colored, 'RGBA')

    buf = io.BytesIO()
    heatmap_img.save(buf, format='PNG')
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    peak = float(heatmap_resized.max())
    avg = float(heatmap_resized.mean())
    focus_pixels = (heatmap_resized > 0.5).sum()
    total_pixels = heatmap_resized.size
    focus_ratio = focus_pixels / total_pixels

    metrics = {
        'peak_attention': f'{peak:.1%}',
        'avg_attention': f'{avg:.1%}',
        'focus_ratio': f'{focus_ratio:.1%}',
    }

    return jsonify({'heatmap': heatmap_b64, 'metrics': metrics})


if __name__ == '__main__':
    print('Loading DeepGaze IIE model...')
    load_model()
    print(f'Model loaded on {DEVICE}')
    print('Server running at http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
