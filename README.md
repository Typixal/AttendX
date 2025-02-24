# AttendX - Face Recognition Attendance System

## Overview
AttendX is a face recognition-based attendance system that captures real-time images, compares them with stored faces, and logs attendance. It is built using OpenCV, DeepFace, Flask, and a simple web interface.

## Features
- Live face recognition using DeepFace
- Capture and store known faces
- Mark attendance automatically
- View and clear attendance logs
- Web interface for easy access

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Flask
- OpenCV
- DeepFace
- Pandas
- NumPy
- Bootstrap (for frontend styling)

### Install Dependencies
Run the following command to install required packages:
```sh
pip install flask opencv-python deepface pandas numpy
```

## Usage
### 1. Start the Application
Run the Flask backend:
```sh
python app.py
```

### 2. Open the Web Interface
Go to:
```
http://127.0.0.1:5000
```

### 3. Capture and Store Known Faces
- Enter a name in the input field.
- Click "Capture Face" to save images to the `known_faces/` directory.

### 4. Start Face Recognition
- Click "Start Recognition" to begin real-time face detection.

### 5. View & Manage Attendance
- The system automatically logs attendance when a face is recognized.
- Click "Refresh Log" to update the table.
- Click "Clear Attendance" to reset logs.

## Folder Structure
```
AttendX/
│── app.py             # Main backend script
│── capture.py         # Captures and saves images
│── templates/
│   ├── index.html     # Frontend UI
│── static/
│   ├── css/           # Stylesheets
│── known_faces/       # Folder to store registered faces
│── attendance.csv     # Attendance log file
│── README.md          # Documentation
```

## Troubleshooting
### 1. No column to parse from file
If you get this error, ensure `attendance.csv` exists and is not empty.
```sh
echo "Name,Time" > attendance.csv
```

### 2. Camera Feed Not Displaying
- Ensure your webcam is working.
- Restart the Flask app and refresh the browser.

### 3. Face Not Recognized
- Ensure clear lighting.
- Capture multiple images for better accuracy.

## License
MIT License

