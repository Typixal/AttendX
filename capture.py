import cv2
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

@app.route('/capture', methods=['POST'])
def capture():
    name = request.form.get("name", "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to access the camera"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    # Save image with the name
    image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(image_path, frame)

    return jsonify({"success": True, "message": f"Image saved as {name}.jpg"}), 200

if __name__ == "__main__":
    app.run(debug=True)
