"""
StreetScan AI: Real-Time Infrastructure Damage Assessment System
Engineered by Azizullah Naik
"""

import os
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

# Class labels for damage detection (YOLOv11 trained classes)
CLASS_LABELS = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

# Color scheme for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 99, 71),    # Tomato Red for Pothole
    1: (50, 205, 50),    # Lime Green for Crack
    2: (70, 130, 180)    # Steel Blue for Manhole
}


def initialize_model():
    """
    Initialize YOLOv11 model with custom best.pt weights using safe relative paths.
    Includes defensive fallback for serverless timeout/memory limits.
    """
    print("[INIT] Initializing YOLOv11 model with custom best.pt weights...")
    print(f"[INIT] Model path: {MODEL_PATH}")
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = YOLO(MODEL_PATH)
        print("[INIT] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLOv11 model: {str(e)}")
        print("[ERROR] Serverless environment may have timed out or hit memory limits during model compilation")
        return None


def calculate_road_safety_index(total_detections, image_area):
    """
    Calculate Road Safety Index Score based on damage density.
    Returns a percentage (0-100) where higher is safer.
    """
    if total_detections == 0:
        return 100.0
    
    # Damage density: detections per million pixels
    density = (total_detections / image_area) * 1_000_000
    
    # Safety score decreases with higher density
    # Base formula: 100 - (density * penalty_factor)
    penalty_factor = 2.0
    safety_score = max(0, min(100, 100 - (density * penalty_factor)))
    
    return round(safety_score, 2)


def draw_detections_opencv(image_np, results):
    """
    Draw bounding boxes with labeled overlays using OpenCV.
    Returns the annotated image as numpy array (BGR format).
    """
    annotated = image_np.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Ensure class_id is within valid range
                if class_id not in CLASS_LABELS:
                    continue
                
                label = CLASS_LABELS[class_id]
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                text = f"{label} {confidence:.1%}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    
    return annotated


def image_to_base64_clean(image_np):
    """
    Convert numpy array (BGR) to clean Base64 encoded string without data URI prefix.
    Uses cv2.imencode for proper JPEG byte array conversion.
    Returns clean Base64 string for frontend consumption.
    """
    # Encode image to JPEG format using OpenCV
    success, encoded_image = cv2.imencode('.jpg', image_np)
    
    if not success:
        raise ValueError("Failed to encode image to JPEG format")
    
    # Convert to bytes and then to clean Base64 (no prefix)
    img_bytes = encoded_image.tobytes()
    img_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_str


# Initialize at startup
model = initialize_model()


@app.route('/')
def index():
    """Render the main dashboard interface."""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_damage():
    """
    Main detection endpoint with real YOLOv11 inference.
    Accepts uploaded image, runs YOLO detection, returns annotated image with statistics.
    Includes defensive fallback for serverless timeout/memory limits.
    """
    try:
        print("[DEBUG] /api/detect endpoint called")
        print(f"[DEBUG] Request files keys: {list(request.files.keys())}")
        
        # Defensive check: Ensure model is loaded
        if model is None:
            print("[ERROR] Model not initialized - serverless environment may have failed during startup")
            return jsonify({
                'error': 'Model initialization failed. Serverless environment may have timed out or exceeded memory limits during model compilation.',
                'success': False
            }), 503
        
        # Handle file upload with fallback keys
        file = request.files.get('image') or request.files.get('file')
        
        if file is None:
            print("[ERROR] No file found in request (checked 'image' and 'file' keys)")
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        if file.filename == '':
            print("[ERROR] File has empty filename")
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        print(f"[DEBUG] File received: {file.filename}, size: {file.content_length}")
        
        # Read file buffer and decode using OpenCV
        file_bytes = file.read()
        print(f"[DEBUG] File bytes read: {len(file_bytes)} bytes")
        
        # Convert to numpy array and decode with OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_np is None:
            print("[ERROR] Failed to decode image with OpenCV")
            return jsonify({
                'error': 'Failed to decode image',
                'success': False
            }), 400
        
        print(f"[DEBUG] Image decoded successfully, shape: {image_np.shape}")
        
        # Get image dimensions for safety calculation
        height, width = image_np.shape[:2]
        image_area = height * width
        print(f"[DEBUG] Image dimensions: {width}x{height}, area: {image_area}")
        
        # Run YOLOv11 inference with timeout protection
        print(f"[DETECT] Running YOLOv11 inference...")
        try:
            results = model(image_np, verbose=False)
        except Exception as inference_error:
            print(f"[ERROR] YOLOv11 inference failed: {str(inference_error)}")
            print("[ERROR] Serverless environment may have timed out during inference")
            return jsonify({
                'error': 'Inference timeout or memory limit exceeded. Please try with a smaller image or contact support.',
                'success': False
            }), 504
        
        # Process detection results
        detections = {
            'Pothole': 0,
            'Crack': 0,
            'Manhole': 0
        }
        
        total_detections = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id in CLASS_LABELS:
                        label = CLASS_LABELS[class_id]
                        detections[label] += 1
                        total_detections += 1
        
        print(f"[DETECT] YOLOv11 detections: {total_detections} objects")
        
        # Draw annotations on image using OpenCV
        annotated_image = draw_detections_opencv(image_np, results)
        print(f"[DETECT] Real detections found: {total_detections} objects")
        
        # Convert to clean Base64 (no data URI prefix)
        base64_image = image_to_base64_clean(annotated_image)
        print(f"[DEBUG] Base64 encoded image length: {len(base64_image)} characters")
        
        # Calculate statistics with dynamic severity based on total damages
        road_safety_index = calculate_road_safety_index(total_detections, image_area)
        
        # Dynamic severity calculation based on total damage count
        if total_detections == 0:
            severity = "LOW"
            road_safety_index = 100.0
        elif total_detections <= 2:
            severity = "LOW"
            road_safety_index = max(75.0, road_safety_index)
        elif total_detections <= 5:
            severity = "MODERATE"
            road_safety_index = max(50.0, min(74.0, road_safety_index))
        else:
            severity = "HIGH"
            road_safety_index = max(25.0, min(49.0, road_safety_index))
        
        # Prepare response with EXACT structure as requested by frontend
        response = {
            "success": True,
            "image": base64_image,
            "counts": {
                "potholes": detections['Pothole'],
                "cracks": detections['Crack'],
                "manholes": detections['Manhole']
            },
            "severity": severity,
            "safety_index": int(road_safety_index)
        }
        
        print(f"[DETECT] Detection complete: {total_detections} objects found, safety_index: {road_safety_index}")
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] Detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'success': False
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring system status."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("StreetScan AI: Real-Time Infrastructure Damage Assessment")
    print("Engineered by Azizullah Naik | AI Engineer Internship Task 4")
    print("=" * 60)
    print("[SERVER] Starting Flask development server...")
    print("[SERVER] Dashboard will be available at: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
