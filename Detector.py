import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
from com_in_ineuron_ai_utils.utils import encodeImageIntoBase64

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from logger import getLog

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
logger=getLog('Detector.py')

class Detector():

    def __init__(self):

        try:

            self.model = tf.saved_model.load("my_model\saved_model")
            self.category_index = label_map_util.create_category_index_from_labelmap("labelmap.pbtxt",use_display_name=True)
            logger.info("Detector object initialized")
        
        except Exception as e:
            
            logger.exception(f"Failed to intialize Detector object : \n{e}")
            raise Exception("Failed to intialize Detector object")
    
    def load_image_into_numpy_array(self, path):

        try:

            img_data = tf.io.gfile.GFile(path, 'rb').read()
            image = Image.open(BytesIO(img_data))
            (im_width, im_height) = image.size
            logger.info("Image loaded into numpy array")
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        
        except Exception as e:

            logger.exception(f"Failed to load image as numpy array : \n{e}")
            raise Exception("Failed to load image as numpy array")
    
    def run_inference_for_single_image(self, model, image):

        try:

            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]
            output_dict = model(input_tensor)
            logger.info("Predictions completed")
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
            output_dict['num_detections'] = num_detections
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
            if 'detection_masks' in output_dict:
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
                output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            logger.info("Inference for single Image completed")
            return output_dict

        except Exception as e:

            logger.exception(f"Failed to run inference for single image : \n{e}")
            raise Exception("Failed to run inference for single image")

    def run_inference(self):

        try:

            image_path = "inputImage.jpg"
            image_np = self.load_image_into_numpy_array(image_path)
            model = self.model
            output_dict = self.run_inference_for_single_image(model, image_np)
            category_index = self.category_index
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
            output_filename = 'output.jpg'
            cv2.imwrite(output_filename, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            logger.info("Output Image Saved")
            opencodedbase64 = encodeImageIntoBase64("output.jpg")
            result = {"image": opencodedbase64.decode('utf-8')}
            logger.info("Inference Completed")
            return result

        except Exception as e:

            logger.exception(f"Failed to complete inference : \n{e}")
            raise Exception("Failed to complete inference ")