import os
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ultralytics import YOLO
from collections import defaultdict
import math
import time

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

# Severity levels for different findings
SEVERITY_LEVELS = {
    "Caries": "high",
    "Periapical lesion": "high",
    "Bone Loss": "high",
    "Fractured teeth": "high",
    "Cyst": "high",
    "Root resorption": "medium",
    "Retained root": "medium",
    "Missing teeth": "medium",
    "Impacted tooth": "medium",
    "Malaligned": "medium",
    "Bone defect": "medium",
    "Root Canal Treatment": "low",
    "Crown": "low",
    "Filling": "low",
    "Implant": "low",
    "Post-core": "low",
    "Abutment": "low",
    "Orthodontic brackets": "low",
    "Metal band": "low",
    "Permanent retainer": "low",
    "Wire": "low",
    "TAD": "low",
    "Gingival former": "low",
    "Plating": "low",
    "Attrition": "low",
    "Supra Eruption": "low",
    "Permanent Teeth": "info",
    "Primary teeth": "info",
    "Mandibular Canal": "info",
    "Maxillary sinus": "info",
    "Root Piece": "info"
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    return response

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_overlap_percentage(box1, box2):
    """Calculate what percentage of box1 is overlapped by box2"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    return intersection / area1 if area1 > 0 else 0

def combine_overlapping_detections(detections, iou_threshold=0.5, overlap_threshold=0.45):
    """
    Combine highly overlapping detections of the same class
    - Use overlap_threshold (60%) for combining similar detections
    - Use iou_threshold for general NMS
    """
    if not detections:
        return []
    
    # Group by class
    class_groups = defaultdict(list)
    for det in detections:
        class_groups[det['label']].append(det)
    
    combined_detections = []
    
    for class_name, dets in class_groups.items():
        # Sort by confidence (highest first)
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while dets:
            # Take the highest confidence detection
            current = dets.pop(0)
            current_box = [
                current['bounding_box']['x1'],
                current['bounding_box']['y1'],
                current['bounding_box']['x2'],
                current['bounding_box']['y2']
            ]
            
            # Check for high overlap with remaining detections
            to_merge = [current]
            remaining = []
            
            for det in dets:
                det_box = [
                    det['bounding_box']['x1'],
                    det['bounding_box']['y1'],
                    det['bounding_box']['x2'],
                    det['bounding_box']['y2']
                ]
                
                overlap = calculate_overlap_percentage(det_box, current_box)
                iou = calculate_iou(current_box, det_box)
                
                # Merge if overlap is very high
                if overlap > overlap_threshold or iou > iou_threshold:
                    to_merge.append(det)
                else:
                    remaining.append(det)
            
            # If multiple detections to merge, create a combined detection
            if len(to_merge) > 1:
                # Average the bounding boxes weighted by confidence
                total_conf = sum(d['confidence'] for d in to_merge)
                avg_x1 = sum(d['bounding_box']['x1'] * d['confidence'] for d in to_merge) / total_conf
                avg_y1 = sum(d['bounding_box']['y1'] * d['confidence'] for d in to_merge) / total_conf
                avg_x2 = sum(d['bounding_box']['x2'] * d['confidence'] for d in to_merge) / total_conf
                avg_y2 = sum(d['bounding_box']['y2'] * d['confidence'] for d in to_merge) / total_conf
                avg_conf = total_conf / len(to_merge)
                
                merged = current.copy()
                merged['bounding_box'] = {
                    'x1': avg_x1,
                    'y1': avg_y1,
                    'x2': avg_x2,
                    'y2': avg_y2
                }
                merged['confidence'] = avg_conf
                merged['merged_count'] = len(to_merge)
                keep.append(merged)
            else:
                current['merged_count'] = 1
                keep.append(current)
            
            dets = remaining
        
        combined_detections.extend(keep)
    
    return combined_detections

def calculate_detection_area(bbox):
    """Calculate area of bounding box"""
    return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])

def analyze_spatial_distribution(detections, img_width, img_height):
    """Analyze spatial distribution of detections"""
    if not detections:
        return {}
    
    # Divide image into quadrants
    quadrants = {
        'top_left': 0,
        'top_right': 0,
        'bottom_left': 0,
        'bottom_right': 0
    }
    
    for det in detections:
        bbox = det['bounding_box']
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2
        
        if center_x < img_width / 2:
            if center_y < img_height / 2:
                quadrants['top_left'] += 1
            else:
                quadrants['bottom_left'] += 1
        else:
            if center_y < img_height / 2:
                quadrants['top_right'] += 1
            else:
                quadrants['bottom_right'] += 1
    
    return quadrants

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Serve the upload page HTML"""
    return send_file('upload_page_advanced.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Run YOLO prediction
    results = model.predict(original_img, conf=0.25)
    
    # Process all detections
    raw_detections = []
    
    for idx, (detection, class_id) in enumerate(zip(results[0].boxes, results[0].boxes.cls)):
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        conf = detection.conf[0].item()
        class_id_int = int(class_id.item())
        class_name = results[0].names[class_id_int]
        color = PALETTE[class_id_int % len(PALETTE)]
        
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
            "segmentation": segmentation,
            "severity": SEVERITY_LEVELS.get(class_name, "info")
        }
        raw_detections.append(detection_data)
    
    # Combine overlapping detections
    combined_detections = combine_overlapping_detections(raw_detections, 
                                                         iou_threshold=0.5, 
                                                         overlap_threshold=0.85)
    
    # Re-assign IDs after combining
    for idx, det in enumerate(combined_detections):
        det['id'] = idx
        det['area'] = calculate_detection_area(det['bounding_box'])
    
    # Calculate statistics
    class_counts = defaultdict(int)
    severity_counts = defaultdict(int)
    
    for det in combined_detections:
        class_counts[det['label']] += 1
        severity_counts[det['severity']] += 1
    
    # Spatial distribution analysis
    spatial_dist = analyze_spatial_distribution(combined_detections, 
                                                original_img.width, 
                                                original_img.height)
    
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
            "count": class_counts.get(name, 0),
            "severity": SEVERITY_LEVELS.get(name, "info")
        })
    
    # Convert original image to base64
    original_b64 = image_to_base64(original_img)
    
    # Calculate average confidence
    avg_confidence = sum(d['confidence'] for d in combined_detections) / len(combined_detections) if combined_detections else 0
    
    # Prepare response
    response_data = {
        "original_image": original_b64,
        "detections": combined_detections,
        "classes": class_list,
        "image_dimensions": {
            "width": original_img.width,
            "height": original_img.height
        },
        "statistics": {
            "total_detections": len(combined_detections),
            "raw_detections": len(raw_detections),
            "combined_count": len(raw_detections) - len(combined_detections),
            "average_confidence": float(avg_confidence),
            "severity_distribution": dict(severity_counts),
            "spatial_distribution": spatial_dist,
            "class_distribution": dict(class_counts)
        }
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


@app.route('/save_export', methods=['POST'])
def save_export():
    """Save exported image posted from client (base64 PNG).
    Expects JSON: { 'image': 'data:image/png;base64,...', 'filename': 'optional.png' }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            _, img_b64 = img_b64.split(',', 1)

        img_bytes = base64.b64decode(img_b64)

        export_dir = os.path.join(HOME, 'exports')
        os.makedirs(export_dir, exist_ok=True)

        filename = data.get('filename') or f'export_{int(time.time())}.png'
        filepath = os.path.join(export_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)

        return jsonify({'saved': True, 'path': filepath}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
