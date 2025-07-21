import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import cv2
import zipfile

from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------------ Verificare fișiere ------------------------

def check_file_exists(path):
    if not os.path.isfile(path):
        print(f"[EROARE] Fisierul NU exista: {path}")
    else:
        print(f"[OK] Fisierul exista: {path}")

def check_keras_valid(path):
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            corrupt = zip_ref.testzip()
            if corrupt:
                print(f"[CORUPT] Fisierul `.keras` este corupt la: {corrupt}")
            else:
                print(f"[VALID] Fisierul `.keras` este valid: {path}")
    except zipfile.BadZipFile:
        print(f"[INVALID] Fișierul NU este un `.keras` valid (zip corupt): {path}")

check_file_exists("simple_model.keras")
check_file_exists("EN_model.keras")
check_file_exists("hybrid_quantum_model.pth")
check_keras_valid("simple_model.keras")
check_keras_valid("EN_model.keras")

# ------------------------ Încărcare modele ------------------------

# Keras models
simple_model = load_model("simple_model.keras")
efficientnet_model = load_model("EN_model.keras")

from hybrid_net import HybridNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model = HybridNet().to(device)
hybrid_model.load_state_dict(torch.load("hybrid_quantum_model.pth", map_location=device))
hybrid_model.eval()

# ------------------------ Funcții utile ------------------------

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) 
    return img, img_array

def keras_predict(model, img_array, preprocess_fn=None):
    x = np.expand_dims(img_array, axis=0)
    if preprocess_fn:
        x = preprocess_fn(x)
    return model.predict(x)

def torch_predict(model, img_pil):
    transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    x = transform(img_pil.convert('L')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

def show_heatmap(img_array, class_idx, model):
    img_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)

    # ACTUALIZEAZA după model.summary() daca este alt layer
    last_conv_layer_name = "conv2d"
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
    heatmap = tf.squeeze(heatmap).numpy()

    return heatmap

# ------------------------ GUI ------------------------

def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img, img_array = preprocess_image(file_path)

    img_display = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    pred1 = keras_predict(simple_model, img_array)  
    pred2 = keras_predict(efficientnet_model, img_array, preprocess_input)
    pred3 = torch_predict(hybrid_model, img)

    label_map = ["Benign", "Malignant", "Normal"]
    out_text = f"""
    Simple CNN: {label_map[np.argmax(pred1)]} ({np.max(pred1):.2f})
    EfficientNet: {label_map[np.argmax(pred2)]} ({np.max(pred2):.2f})
    Quantum Hybrid: {label_map[np.argmax(pred3)]} ({np.max(pred3):.2f})
    """
    result_label.config(text=out_text)

    # Heatmap
    heatmap = show_heatmap(img_array, np.argmax(pred1), simple_model)
    overlay = np.array(img.resize((224, 224)).convert("RGB"))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)

    # Afisare heatmap în GUI
    heatmap_img = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    heatmap_img = heatmap_img.resize((300, 300))
    heatmap_tk = ImageTk.PhotoImage(heatmap_img)
    heatmap_label.config(image=heatmap_tk)
    heatmap_label.image = heatmap_tk

# ------------------------ Interfață ------------------------

root = tk.Tk()
root.title("Clasificator CT tumori pulmonare")
root.geometry("600x750")

btn = tk.Button(root, text=" Încarcă imagine", command=load_image)
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=5)

result_label = tk.Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

heatmap_label = tk.Label(root)
heatmap_label.pack(pady=10)

root.mainloop()
