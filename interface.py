import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow.lite as tflite
from PIL import Image, ImageTk

# Automatically detect script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "emotion_model.tflite")

# Load the TFLite model with error handling
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
    exit()

# Emotion labels
EMOTIONS = ["angry", "disgust", "Fear", "Happy", "neutral", "sad", "surprise"]

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Image preprocessing function
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

# Emotion prediction function
def predict_emotion(img):
    try:
        img = preprocess_image(img)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return EMOTIONS[np.argmax(output_data)]
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
        return "Unknown"

# Open image from file
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to load the image.")
        return
    display_image(img)
    emotion = predict_emotion(img)
    result_label.config(text=f"Detected Emotion: {emotion}")

# Start camera thread
def open_camera():
    threading.Thread(target=camera_thread, daemon=True).start()

# Camera processing function
def camera_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access the camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if roi.shape[0] >= 48 and roi.shape[1] >= 48:
                emotion = predict_emotion(roi)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Camera - Emotion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Display image in GUI
def display_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    image_panel.config(image=img_tk)
    image_panel.image = img_tk

# Show developer info
def show_about():
    messagebox.showinfo("About", "Developer: Houssem Bouagal\nEmail: mouhamedhoussem813@gmail.com")

# Close application
def quit_app():
    root.quit()

# Tkinter GUI setup
root = tk.Tk()
root.title("Emotion Recognition System")
root.geometry("500x250")
root.configure(bg="#f8f9fa")

# Set window icon
icon_path = os.path.join(script_dir, "emotion-recognition.png")
if os.path.exists(icon_path):
    root.iconphoto(False, tk.PhotoImage(file=icon_path))

# Header label
header = tk.Label(root, text="Emotion Recognition System", font=("Arial", 16, "bold"), bg="#007bff", fg="white", pady=5)
header.pack(fill=tk.X)

# Buttons frame
button_frame = tk.Frame(root, bg="#f8f9fa")
button_frame.pack(pady=15)

btn_import = tk.Button(button_frame, text="Test an Image", command=open_image, font=("Arial", 10, "bold"), bg="#007bff", fg="white", width=18)
btn_import.grid(row=0, column=0, padx=10, pady=5)

btn_camera = tk.Button(button_frame, text="Start Camera", command=open_camera, font=("Arial", 10, "bold"), bg="#28a745", fg="white", width=18)
btn_camera.grid(row=0, column=1, padx=10, pady=5)

btn_about = tk.Button(button_frame, text="About", command=show_about, font=("Arial", 10, "bold"), bg="#17a2b8", fg="white", width=18)
btn_about.grid(row=1, column=0, padx=10, pady=5)

btn_quit = tk.Button(button_frame, text="Quit", command=quit_app, font=("Arial", 10, "bold"), bg="#dc3545", fg="white", width=18)
btn_quit.grid(row=1, column=1, padx=10, pady=5)

# Emotion detection result label
result_label = tk.Label(root, text="", font=("Arial", 12), bg="#f8f9fa", fg="black")
result_label.pack(pady=5)

# Image display panel
image_panel = tk.Label(root, bg="#f8f9fa", borderwidth=0, highlightthickness=0, width=200, height=200)

image_panel.pack(pady=10)

# Keyboard shortcuts
root.bind("<Control-o>", lambda event: open_image())
root.bind("<Control-q>", lambda event: quit_app())

# Run the application
root.mainloop()
