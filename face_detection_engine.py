''' module face_detection_engine.py

    Purpose: implement a face detecting engine
'''

from enum import Enum

from numpy import asarray
from PIL import Image

from face_embedding_engine import FaceEmbeddingModelEnum

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

SSD_MOBILENET_V2_FACE_MODEL = 'models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite' # trained with Celebrity imageset

class FaceDetectionMethodEnum(Enum):
    ''' enum DetectionMethod

        Enumerates all methods supported for detecting faces
    '''
    MTCNN = 1 # currently only supported on Ubuntu
    SSD_MOBILENET_V2 = 2 # currently only supported on Coral dev board

class FaceDetectionEngine:
    ''' class FaceDetectionEngine

        Purpose: detect faces in an image
    '''

    def __init__(self, detection_method):
        ''' function constructor

        Constructor for FaceDetectionEngine

        Args:
            detection_method (DetectionMethod): Method to use for detection

        Returns:
            None
        '''

        # We only want to import these modules at run-time since
        # they will only be installed on certain platforms.
        # pylint: disable=import-outside-toplevel, import-error

        self.detection_method = detection_method
        if self.detection_method == FaceDetectionMethodEnum.MTCNN:
            # create the MTCNN detector, using default weights
            print("Using MTCNN for face detection")
            from mtcnn.mtcnn import MTCNN
            self.face_detection_engine = MTCNN()
        elif self.detection_method == FaceDetectionMethodEnum.SSD_MOBILENET_V2:
            # load the MobileNet V2 SSD Face model
            print("Using SSD MobileNet V2 for face detection")
            from edgetpu.detection.engine import DetectionEngine
            self.face_detection_engine = DetectionEngine(SSD_MOBILENET_V2_FACE_MODEL)
        else:
            raise Exception("Invalid detection method: {}".format(detection_method))

    # detect faces
    def detect_faces(self, rgb_array):
        ''' function detect_faces

        Detect any faces that are present in the given image.

        Args:
            rgb_array (numpy.ndarray): An image that may or may not contain faces

        Returns:
            An array of bounding boxes (top_left_x, top_left_y, width, height)
            for each face detected in the given image
        '''

        results = [] # assume no faces are detected

        # detect faces in the image
        if self.detection_method == FaceDetectionMethodEnum.MTCNN:
            detected_faces = self.face_detection_engine.detect_faces(rgb_array)

            # extract the bounding box from the first face
            if len(detected_faces) == 0:
                return results

            for detected_face in detected_faces:
                # note the bounding box is in the format we want
                results.append(tuple(detected_face['box']))

        else: # DetectionMethod.SSD_MOBILENET_V2
            frame_as_image = Image.fromarray(rgb_array)
            detected_faces = self.face_detection_engine.detect_with_image(
                frame_as_image,
                threshold=0.5,
                keep_aspect_ratio=True,
                relative_coord=False,
                top_k=5,
                resample=Image.BOX)

            if len(detected_faces) == 0:
                return results

            # extract the bounding box from the first face
            for detected_face in detected_faces:
                # convert the bounding box to the format we want
                x_1, y_1, x_2, y_2 = detected_face.bounding_box.flatten().astype("int")
                width = abs(x_2 - x_1)
                height = abs(y_2 - y_1)
                result = (x_1, y_1, width, height)
                results.append(result)

        return results

    def extract_face(self, rgb_array, embedding_model):
        ''' function extract_face

        Extract a single face from the given frame

        Args:
            rgb_array (numpy.ndarray): The image that may or may not contain
                            one or more faces
            embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                            embeddings for face images

        Returns:
            If a face is detected, returns an RGB numpy.ndarray of the face extracted from
            the given frame of the dimensions required for the given embedding model.
            Otherwise it returns an empty array.
        '''

        detected_faces = self.detect_faces(rgb_array)
        if len(detected_faces) == 0:
            return []

        if detected_faces[0][2] == 0:
            return []

        x_1, y_1, width, height = tuple(detected_faces[0])
        x_1, y_1 = abs(x_1), abs(y_1)
        x_2, y_2 = x_1 + width, y_1 + height

        # extract a cropped image of the detected face
        face = rgb_array[y_1:y_2, x_1:x_2]

        # resize pixels to the dimension required for the specified embedding model
        image = Image.fromarray(face)
        image = image.resize((160, 160))

        # convert image to numpy array
        face_rgb_array = asarray(image)
        return face_rgb_array
