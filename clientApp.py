from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage
from Detector import Detector
from object_detection.utils import label_map_util

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.obj_detect = Detector()

#@app.route("/")
#def home():
#    return render_template("index.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.obj_detect.run_inference()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    port = 8000
    app.run(host='127.0.0.1', port=port)