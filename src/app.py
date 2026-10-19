"""Tkinter desktop app: classify an uploaded image or start live camera detection.

Usage:
    python -m src.app
"""
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from src import realtime
from src.config import ASSETS_DIR
from src.model import load_trained_model
from src.predict import predict_image

BG_COLOR = "#3c6e71"

model = load_trained_model()


def load_icon(filename, size=(80, 80)):
    icon = Image.open(ASSETS_DIR / filename).resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(icon)


def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
    if not file_path:
        return

    # Show the selected image
    preview = ImageTk.PhotoImage(Image.open(file_path).resize((300, 300), Image.LANCZOS))
    panel.config(image=preview)
    panel.image = preview  # keep a reference so it is not garbage collected

    # Show the predicted class
    class_name, confidence = predict_image(model, file_path)
    predicted_label.config(text=f"Predicted class: {class_name} ({confidence:.0%})")
    predicted_label.place(x=950, y=270)


def start_live_detection():
    realtime.run(model)


root = tk.Tk()
root.title("Waste classification")
root.geometry("1200x509")

# Background
background_photo = ImageTk.PhotoImage(Image.open(ASSETS_DIR / "background.png"))
tk.Label(root, image=background_photo).place(relwidth=1, relheight=1)

# Title
tk.Label(root, text="Waste classification", font=("Helvetica", 48, "bold"),
         bg=BG_COLOR, fg="yellow").pack(pady=20)

# Live detection button
live_photo = load_icon("camera_icon.png")
tk.Button(root, image=live_photo, command=start_live_detection).place(x=120, y=150)
tk.Label(root, text="Live detection", font=("Arial", 12),
         bg=BG_COLOR, fg="yellow").place(x=110, y=230)

# Upload button
upload_photo = load_icon("upload_icon.png")
tk.Button(root, image=upload_photo, command=upload_image).place(x=320, y=150)
tk.Label(root, text="Upload an image", font=("Arial", 12),
         bg=BG_COLOR, fg="yellow").place(x=300, y=230)

# Image preview and prediction
panel = tk.Label(root, bg=BG_COLOR)
panel.place(x=600, y=150)
predicted_label = tk.Label(root, text="", font=("Arial", 16), bg=BG_COLOR, fg="white")

root.mainloop()
