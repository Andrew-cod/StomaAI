import os
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ultralytics import YOLO

HOME = os.getcwd()
model = YOLO(f'{HOME}/best.pt')

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (30, 144, 255), (0, 191, 255), (135, 206, 250), (70, 130, 180), (123, 104, 238),
    (72, 61, 139), (138, 43, 226), (148, 0, 211), (186, 85, 211), (255, 20, 147),
    (255, 105, 180), (255, 160, 122), (255, 69, 0), (255, 99, 71), (218, 112, 214),
    (238, 130, 238), (255, 222, 173)
]

CLASS_NAMES = [
    "Caries", "Crown", "Filling", "Implant", "Malaligned", "Mandibular Canal", "Missing teeth",
    "Periapical lesion", "Retained root", "Root Canal Treatment", "Root Piece", "Impacted tooth",
    "Maxillary sinus", "Bone Loss", "Fractured teeth", "Permanent Teeth", "Supra Eruption", "TAD",
    "Abutment", "Attrition", "Bone defect", "Gingival former", "Metal band", "Orthodontic brackets",
    "Permanent retainer", "Post-core", "Plating", "Wire", "Cyst", "Root resorption", "Primary teeth"
]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    return response

def get_optimal_font_size(img_width, img_height):
    """Calculate optimal font size based on image dimensions"""
    base_size = min(img_width, img_height) // 40
    return max(12, min(base_size, 24))

def draw_rounded_rectangle(draw, coords, radius, fill, outline=None, width=1):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = coords
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill, outline=outline, width=width)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill, outline=outline, width=width)
    draw.pieslice([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, fill=fill, outline=outline)
    draw.pieslice([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, fill=fill, outline=outline)
    draw.pieslice([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, fill=fill, outline=outline)
    draw.pieslice([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, fill=fill, outline=outline)

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Serve the upload page HTML"""
    return send_file('upload_page_professional.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Try to load fonts
        try:
            font_size = get_optimal_font_size(original_img.width, original_img.height)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 2)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Run YOLO prediction
    results = model.predict(original_img, conf=0.25)
    
    # Process all detections
    detections = []
    class_counts = {}
    
    for idx, (detection, class_id) in enumerate(zip(results[0].boxes, results[0].boxes.cls)):
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        conf = detection.conf[0].item()
        class_id_int = int(class_id.item())
        class_name = results[0].names[class_id_int]
        color = PALETTE[class_id_int % len(PALETTE)]
        
        # Count detections
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Handle segmentation if available
        segmentation = None
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            try:
                seg = results[0].masks.xy[idx]
                segmentation = [{"x": float(x), "y": float(y)} for x, y in seg]
            except:
                pass
        
        detection_data = {
            "id": idx,
            "label": class_name,
            "confidence": float(conf),
            "bounding_box": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            },
            "color": {
                "r": color[0],
                "g": color[1],
                "b": color[2]
            },
            "segmentation": segmentation
        }
        detections.append(detection_data)
    
    # Create class list with colors and counts
    class_list = []
    for i, name in enumerate(CLASS_NAMES):
        color = PALETTE[i % len(PALETTE)]
        class_list.append({
            "name": name,
            "color": {
                "r": color[0],
                "g": color[1],
                "b": color[2]
            },
            "count": class_counts.get(name, 0)
        })
    
    # Convert original image to base64
    original_b64 = image_to_base64(original_img)
    
    # Prepare response
    response_data = {
        "original_image": original_b64,
        "detections": detections,
        "classes": class_list,
        "image_dimensions": {
            "width": original_img.width,
            "height": original_img.height
        },
        "total_detections": len(detections)
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
