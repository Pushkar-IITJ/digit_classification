from flask import Flask, request,jsonify
from joblib import load

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

    predicted_digit = model.predict([image])[0]

    return jsonify({"digit": predicted_digit})

if __name__ == "__main__":
    app.run()

