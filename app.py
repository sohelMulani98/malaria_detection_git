from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "malaria_detection.h5")

# Load model
model = load_model(MODEL_PATH)

# Image size expected by the model
IMG_SIZE = (64, 64)  # Change if your model expects a different size

def prepare_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize if model trained that way
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded.", 400
        file = request.files["file"]
        if file.filename == "":
            return "Empty filename.", 400
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            img = prepare_image(filepath)
            prediction = model.predict(img)[0][0]
            result = "Uninfected" if prediction >= 0.5 else "Parasitized"
            confidence = round(prediction * 100, 2) if result == "Uninfected" else round((1 - prediction) * 100, 2)

            return render_template("index.html", result=result, confidence=confidence, image_path=filepath)

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
