from flask import Flask, request,jsonify

app = Flask(__name__)

@app.route("/")
def helloworld():
    return "Hello World!"

# Q4 solution:
import joblib
from flask import request, jsonify

def load_model():
    models = {}
    models['svm'] = joblib.load('models/svm_gamma:0.001_C:10.joblib')
    models['lr'] = joblib.load('models/M22AIE213_lr_lbfgs.joblib')
    models['tree'] = joblib.load('models/tree_max_depth:15.joblib')
    return models

models = load_model()


@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    data = request.json
    model = models.get(model_type)

    if not model:
        return jsonify({'error': 'Model not found'}), 404

    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})





# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     if data.get("image1") == data.get("image2"):
#         result = True
#     else:
#         result = False

#     return jsonify({"result": result})

if __name__ == "__main__":
    app.run()