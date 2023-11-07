from flask import Flask, request,jsonify

app = Flask(__name__)
@app.route("/")
def helloworld():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data.get("image1") == data.get("image2"):
        result = True
    else:
        result = False

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()

