from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import uuid
from ultralytics import YOLO
import re
import subprocess
import time

app = Flask(__name__)
UPLOAD_FOLDER = "static/outputs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO("weights/best.pt")

def reencode_for_web(input_path, output_path):
    """Re-encode video for web compatibility using FFmpeg"""
    cmd = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'fast',
        '-crf', '23',
        '-movflags', '+faststart',  # Moves metadata to beginning
        '-y', output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ FFmpeg re-encoding failed")
        return False

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# Handle image/video upload
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    original_name = os.path.splitext(file.filename)[0]
    safe_name = re.sub(r'[^\w\-_.]', '_', original_name)  # keep letters, numbers, dash, underscore, dot
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}_{safe_name}{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Process image
        results = model(path)
        output_img = results[0].plot()
        
        # Create output filename more explicitly
        name, ext = os.path.splitext(filename)
        output_filename = name + "_detected" + ext
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        
        cv2.imwrite(output_path, output_img)

        # Normalize filename for Flask
        result_file = os.path.basename(output_path).replace("\\", "/")

        # Send to template
        return render_template("index.html", result_file=result_file, file_type="image")

    elif filename.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        # Ensure output folder exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        # Clean output filename
        name, ext = os.path.splitext(filename)
        temp_filename = f"{name}_temp{ext}"
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
        
        output_filename = f"{name}_detected{ext}"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

        # Ensure dimensions are even (required for many codecs)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        # Debug info
        print(f"Input video: {width}x{height}, FPS: {fps}")
        print("Upload folder exists:", os.path.exists(app.config["UPLOAD_FOLDER"]))
        
        # Save using XVID codec (more universally supported than H.264)
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        if not out.isOpened():
            print("❌ VideoWriter failed to open. Check codec and path.")
            return "Error: Could not write output video."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            out.write(results[0].plot())

        cap.release()
        out.release()
        time.sleep(1)  # Flush I/O

        # Re-encode for web compatibility
        if reencode_for_web(temp_path, output_path):
            # Remove temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            print("✅ Video re-encoded for web compatibility")
        else:
            # If re-encoding fails, use original file
            os.rename(temp_path, output_path)
            print("⚠️ Using original encoding (FFmpeg not available)")

        print("✅ Video saved at:", output_path)
        print("✅ File exists:", os.path.exists(output_path))

        result_file = os.path.basename(output_path)
        return render_template("index.html", result_file=result_file, file_type="video")

    else:
        return "Unsupported file format"

# Serve result file
@app.route("/static/outputs/<filename>")
def send_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], 
        filename,
        mimetype='video/mp4' if filename.lower().endswith('.mp4') else None
    )

if __name__ == "__main__":
    app.run(debug=True)