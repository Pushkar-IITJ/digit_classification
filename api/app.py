from flask import Flask, request,jsonify
from joblib import load
import numpy as np

MODEL_PATH = r"models/svm_best.joblib"
model = load(MODEL_PATH)

app = Flask(__name__)
@app.route("/")
def helloworld():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    image = data.get("image")
    processed_image = preprocess_image(image)

    predicted_digit = model.predict([processed_image])[0]

    return jsonify({"digit": predicted_digit})

def preprocess_image(image_data):
    processed_image = np.array(image_data).reshape(1, -1)
    return processed_image

if __name__ == "__main__":
    app.run()

