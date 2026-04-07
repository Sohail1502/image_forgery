"""
Flask Web Application for MobForge-Net
Allows users to upload images and detect forged regions via web interface.

Usage:
  python app.py --weights checkpoints/best_model.pth --port 5000
  
Then open browser: http://localhost:5000
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory
import torchvision.transforms as T

# Add model directory to path
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), 'model')))
from model.mobforge_net import MobForgeNet


# ─────────────────────────────────────────────
# Flask App Setup
# ─────────────────────────────────────────────
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
MODEL_WEIGHTS_PATH = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(weights_path):
    """Load pre-trained model."""
    global MODEL
    print(f"Loading model from {weights_path}...")
    
    if weights_path and os.path.exists(weights_path):
        model = MobForgeNet(pretrained=False).to(DEVICE)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded from checkpoint: {weights_path}")
    else:
        # Load without pre-training (ImageNet weights)
        model = MobForgeNet(pretrained=True).to(DEVICE)
        print("✓ Model loaded with ImageNet pre-training (no saved weights)")
    
    model.eval()
    return model


def preprocess_image(image, img_size=256):
    """Convert PIL image to normalized tensor."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    return tensor


def run_inference(image_path, img_size=256):
    """Run MobForge-Net inference on image."""
    if MODEL is None:
        return None, None, "Model not loaded"
    
    try:
        # Load image
        original_img = Image.open(image_path).convert('RGB')
        original_size = original_img.size  # (W, H)
        
        # Preprocess
        tensor = preprocess_image(original_img, img_size)
        
        # Run inference
        start = time.perf_counter()
        with torch.no_grad():
            pred = MODEL(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Extract probability map
        prob_map = pred[0, 0].cpu().numpy()  # [H, W]
        
        # Resize to original size
        prob_map_resized = Image.fromarray((prob_map * 255).astype(np.uint8))
        prob_map_resized = np.array(prob_map_resized.resize(original_size, Image.BILINEAR))
        prob_map_resized = prob_map_resized.astype(np.float32) / 255.0
        
        # Compute forgery percentage
        forgery_pct = (prob_map_resized > 0.5).mean() * 100
        
        return {
            'prob_map': prob_map_resized,
            'inference_ms': elapsed_ms,
            'forgery_pct': forgery_pct,
            'original_img': original_img,
            'success': True
        }
    except Exception as e:
        return None, str(e), False


def create_visualization(original_img, prob_map):
    """Create side-by-side visualization with heatmap overlay and binary mask."""
    orig_w, orig_h = original_img.size
    target_h = 300
    scale = target_h / orig_h
    target_w = int(orig_w * scale)
    
    # Resize original
    orig_resized = original_img.resize((target_w, target_h), Image.LANCZOS)
    
    # Prepare probability map
    prob_resized = Image.fromarray((prob_map * 255).astype(np.uint8))
    prob_resized = np.array(prob_resized.resize((target_w, target_h), Image.BILINEAR))
    
    # Create heatmap overlay (red channel for forged regions)
    heatmap = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    heatmap[:, :, 0] = prob_resized  # Red channel shows forged regions
    heatmap_img = Image.fromarray(heatmap)
    
    # Blend original with heatmap
    orig_arr = np.array(orig_resized)
    heatmap_arr = np.array(heatmap_img)
    overlay_arr = (0.6 * orig_arr + 0.4 * heatmap_arr).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_arr)
    
    # Create binary mask
    binary_arr = (prob_resized > 127).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary_arr).convert('RGB')
    
    # Compose: Original | Heatmap | Mask
    padding = 15
    total_w = target_w * 3 + padding * 4
    total_h = target_h + 80
    canvas = Image.new('RGB', (total_w, total_h), color=(25, 25, 25))
    
    panels = [orig_resized, overlay_img, binary_img]
    labels = ['Original', 'Forgery Heatmap', 'Binary Mask']
    colors = [(200, 200, 200), (255, 100, 0), (100, 200, 100)]
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    
    x_positions = [padding, target_w + padding*2, target_w*2 + padding*3]
    for x, panel, label, color in zip(x_positions, panels, labels, colors):
        canvas.paste(panel, (x, padding))
        try:
            # Try to use a nice font
            draw.text((x + target_w // 2, target_h + padding + 15),
                     label, fill=color, anchor='mt')
        except:
            # Fallback to default font
            draw.text((x + target_w // 2, target_h + padding + 15),
                     label, fill=color)
    
    return canvas


def img_to_base64(pil_img):
    """Convert PIL image to base64 string."""
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG', quality=95)
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and run inference."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file temporarily
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run inference
        result = run_inference(filepath)
        
        if not result or not result.get('success'):
            return jsonify({'error': result if isinstance(result, str) else 'Inference failed'}), 500
        
        # Create visualization
        viz_img = create_visualization(result['original_img'], result['prob_map'])
        viz_base64 = img_to_base64(viz_img)
        
        # Prepare response
        response = {
            'success': True,
            'inference_ms': round(result['inference_ms'], 2),
            'forgery_pct': round(result['forgery_pct'], 2),
            'visualization': viz_base64,
            'verdict': 'FORGED' if result['forgery_pct'] > 5 else 'AUTHENTIC',
            'device': str(DEVICE),
            'model_params': sum(p.numel() for p in MODEL.parameters()) / 1e6
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Return model status."""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    return jsonify({
        'status': 'ready',
        'device': str(DEVICE),
        'model_loaded': True,
        'weights': MODEL_WEIGHTS_PATH or 'ImageNet pre-training'
    })


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobForge-Net Web Application')
    parser.add_argument('--weights', default=None,
                       help='Path to saved model weights (.pth)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MobForge-Net Web Application")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Load model
    MODEL_WEIGHTS_PATH = args.weights
    MODEL = load_model(args.weights)
    
    print(f"\n🚀 Starting web server on {args.host}:{args.port}")
    print(f"📂 Open browser: http://localhost:{args.port}")
    print("="*70 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False)
