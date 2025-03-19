import tkinter as tk
import numpy as np
import cv2
import joblib
from PIL import Image, ImageGrab

# Load the trained model (update with your actual model file path)
model = joblib.load("model.pkl")

def predict_digit(image):
    # Resize and convert to grayscale
    image = image.resize((28, 28)).convert("L")
    # Invert the image (since the drawn digit might be black on white)
    image = np.invert(np.array(image))
    # Flatten the image to match the input shape of the model
    image = image.reshape(1, 784)
    # Predict using the loaded model
    prediction = model.predict(image)
    return prediction[0]

def draw(event):
    x, y = event.x, event.y
    brush_size = 20  # Adjust the brush size as needed
    canvas.create_oval(x - brush_size/2, y - brush_size/2, x + brush_size/2, y + brush_size/2, fill="black", width=0)


def clear_canvas():
    canvas.delete("all")

def capture_and_predict():
    # Capture canvas and convert to image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Grab the canvas content
    img = ImageGrab.grab((x, y, x1, y1))
    digit = predict_digit(img)

    result_label.config(text=f"Prediction: {digit}")

# Set up the GUI
root = tk.Tk()
root.title("Digit Predictor")
root.geometry("300x400")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

canvas.bind("<B1-Motion>", draw)

btn_frame = tk.Frame(root)
btn_frame.pack()

clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas)
clear_btn.pack(side="left", padx=10, pady=10)

predict_btn = tk.Button(btn_frame, text="Predict", command=capture_and_predict)
predict_btn.pack(side="right", padx=10, pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 20))
result_label.pack()

root.mainloop()
