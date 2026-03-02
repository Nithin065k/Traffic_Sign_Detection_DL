from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# ===== Load trained model =====
# Make sure you saved it as model.save("traffic_classifier.h5") after training
MODEL_PATH = "traffic_classifier.h5"
model = load_model(MODEL_PATH)

# ===== GTSRB class names (43 classes) =====
CLASS_NAMES = [
    "Speed limit (20km/h)",                 # 0
    "Speed limit (30km/h)",                 # 1
    "Speed limit (50km/h)",                 # 2
    "Speed limit (60km/h)",                 # 3
    "Speed limit (70km/h)",                 # 4
    "Speed limit (80km/h)",                 # 5
    "End of speed limit (80km/h)",          # 6
    "Speed limit (100km/h)",                # 7
    "Speed limit (120km/h)",                # 8
    "No passing",                           # 9
    "No passing for vehicles >3.5t",        # 10
    "Right-of-way at next intersection",    # 11
    "Priority road",                        # 12
    "Yield",                                # 13
    "Stop",                                 # 14
    "No vehicles",                          # 15
    "Vehicles >3.5t prohibited",            # 16
    "No entry",                             # 17
    "General caution",                      # 18
    "Dangerous curve to the left",          # 19
    "Dangerous curve to the right",         # 20
    "Double curve",                         # 21
    "Bumpy road",                           # 22
    "Slippery road",                        # 23
    "Road narrows on the right",            # 24
    "Road work",                            # 25
    "Traffic signals",                      # 26
    "Pedestrians",                          # 27
    "Children crossing",                    # 28
    "Bicycles crossing",                    # 29
    "Beware of ice/snow",                   # 30
    "Wild animals crossing",                # 31
    "End of all speed and passing limits",  # 32
    "Turn right ahead",                     # 33
    "Turn left ahead",                      # 34
    "Ahead only",                           # 35
    "Go straight or right",                 # 36
    "Go straight or left",                  # 37
    "Keep right",                           # 38
    "Keep left",                            # 39
    "Roundabout mandatory",                 # 40
    "End of no passing",                    # 41
    "End of no passing >3.5t"               # 42
]

# ===== Image preprocessing to match your training (30x30 RGB, /255) =====
def preprocess_image(file_storage, target_size=(30, 30)):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 30, 30, 3)
    return arr, img  # return both array and PIL image for preview

# ===== Routes =====
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", predicted_label=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please upload an image.")

    file = request.files["file"]

    try:
        x, pil_img = preprocess_image(file, target_size=(30, 30))
        probs = model.predict(x, verbose=0)[0]         # shape: (43,)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100.0
        label = CLASS_NAMES[pred_idx]

        # Show uploaded image on the page without saving to disk (base64)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_b64}"

        # Optionally compute top-3 for debugging/analysis (shown if you want)
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(CLASS_NAMES[int(i)], float(probs[i]) * 100.0) for i in top3_idx]

        return render_template(
            "index.html",
            predicted_label=label,
            confidence=confidence,
            image_data=img_data_uri,
            top3=top3
        )
    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)