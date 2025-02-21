from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import threading

app = Flask(__name__)

# Load dataset
dataset_path = "D:/Ami/Python Projects/AttendX/known_faces/"
student_data = {
    os.path.splitext(file)[0]: os.path.join(dataset_path, file)
    for file in os.listdir(dataset_path)
    if file.endswith(("jpg", "png", "jpeg"))
}

attendance_file = "attendance.csv"

# Global variables for camera control
video_capture = None
face_recognition_running = False


def mark_attendance(name):
    """Mark attendance in CSV, handling empty/missing files."""

    # Handle missing or corrupted CSV file
    try:
        df = pd.read_csv(attendance_file)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame(columns=["Name", "Time"])  # Initialize empty DataFrame
        df.to_csv(attendance_file, index=False)  # Ensure the file is created with headers

    # Check if the name is already in the attendance list
    if name not in df["Name"].values:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([[name, now]], columns=["Name", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)  # Append new entry
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for: {name}")

def recognize_faces():
    """Runs face recognition continuously when enabled."""
    global video_capture, face_recognition_running

    video_capture = cv2.VideoCapture(0)
    while face_recognition_running:
        ret, frame = video_capture.read()
        if not ret:
            break

        try:
            for student_name, student_image_path in student_data.items():
                result = DeepFace.verify(frame, student_image_path, model_name="VGG-Face", enforce_detection=False)

                if result["verified"]:
                    mark_attendance(student_name)
                    cv2.putText(frame, student_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error: {e}")

    video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Generates a video feed for the frontend."""
    def generate():
        global video_capture
        while face_recognition_running:
            if video_capture is None or not video_capture.isOpened():
                continue

            success, frame = video_capture.read()
            if not success:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    """Starts the face recognition process."""
    global face_recognition_running
    if not face_recognition_running:
        face_recognition_running = True
        threading.Thread(target=recognize_faces, daemon=True).start()
    return jsonify({"status": "Recognition started"})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    """Stops the face recognition process."""
    global face_recognition_running
    face_recognition_running = False
    return jsonify({"status": "Recognition stopped"})
@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    """Clears the attendance CSV file."""
    open("attendance.csv", "w").close()
    return "Attendance log cleared successfully", 200

@app.route('/attendance_log', methods=['GET'])
def attendance_log():
    """Fetches attendance data as JSON."""
    if not os.path.exists(attendance_file):
        return jsonify([])
    df = pd.read_csv(attendance_file)
    return jsonify(df.to_dict(orient="records"))

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

@app.route('/capture', methods=['POST'])
def capture():
    name = request.form.get("name", "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to access the camera"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(image_path, frame)

    return jsonify({"success": True, "message": f"Image saved as {name}.jpg"}), 200

if __name__ == '__main__':
    app.run(debug=True)
