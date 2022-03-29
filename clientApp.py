from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage
from Detector import Detector
from object_detection.utils import label_map_util
from logger import getLog

app = Flask(__name__)
CORS(app)

logger=getLog('clientApp.py')
class ClientApp:

    def __init__(self):

        try:

            self.filename = "inputImage.jpg"
            self.obj_detect = Detector()
            logger.info("ClientApp object initialized")

        except Exception as e:

            logger.exception(f"Failed to initialize App Object : \n{e}")
            raise Exception("Failed to initialize App Object")

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():

    try:

        image = request.json['image']
        logger.info("Image loaded")
        decodeImage(image, clApp.filename)
        result = clApp.obj_detect.run_inference()
        logger.info("Prediction Completed")
        return jsonify(result)

    except Exception as e:

        return jsonify(e)

if __name__ == "__main__":
    
    clApp = ClientApp()
    port = 8000
    app.run(host='127.0.0.1', port=port)