''' module face_detector.py

    Purpose: detect faces in an image
'''
import time

from PIL import Image, ImageDraw, ImageFont

from face_detection_engine import FaceDetectionEngine
from face_embedding_engine import get_image_dimensions_for_embedding_model
from face_recognizer import FaceRecognizer
from utils import PRINT_PERFORMANCE_INFO

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

class FaceDetector():
    ''' class FaceDetector

        Purpose: detect (and locate if present) faces in an image
    '''

    def __init__(self, detection_method, embedding_model):
        ''' function constructor

        Constructor for FaceDetector

        Args:
            detection_method (DetectionMethod): Method to use for detection
            embedding_model (FaceEmbeddingModelEnum): The model to use for generating
                            embeddings for face images

        Returns:
            None
        '''

        # load face detection engine
        self.face_detection_engine = FaceDetectionEngine(detection_method)
        self.face_recognizer = FaceRecognizer(embedding_model)
        self.embedding_image_dimensions = get_image_dimensions_for_embedding_model(embedding_model)

        self.start_time_stamp = None
        self.fps_font = ImageFont.truetype(font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=20)
        self.face_label_font = ImageFont.truetype(font="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=20)

    # With the input frame, use the EdgeTPU Detection engine along with the
    # tflite model to detect any faces. If any faces are detected the image
    # will be updated with boxes drawn around each identified face.
    # Note that the frame_as_image object is updated itself.
    def identify_faces_in_frame(self, rgb_array, detect_only=False):
        ''' function identify_faces_in_frame

        Detect any faces that are present in the given image.
        For each detected face, call FaceRecognizer to try to
        identfy it, then draw a box around the face including
        an identification label (or "Unknown" if the face was
        not identified).

        Args:
            rgb_array (numpy.ndarray): The frame that we should try to detect
                            faces in.

        Returns:
            A PIL Image with an enclosing red box and label for each face
            detected in the given frame
        '''

        # record start of main ML processing
        self.start_time_stamp = time.monotonic()

        # Delegate the face detection to the face detection engine
        detection_start_time = time.monotonic()
        detected_faces = self.face_detection_engine.detect_faces(rgb_array)
        detection_end_time = time.monotonic()
        if PRINT_PERFORMANCE_INFO:
            print("Face detection time: {:.3f}s".format(detection_end_time - detection_start_time))

        # convert to a PIL Image
        frame_as_image = Image.fromarray(rgb_array)
        # draw a box drawn around each face detected
        self.draw_face_boxes(frame_as_image, detected_faces, detect_only)

        return frame_as_image

    # draw boxes around each identified face in the image
    def draw_face_boxes(self, frame_as_image, detected_faces, detect_only=False):
        ''' function draw_face_boxes

        For each detected face, try to identify it, then draw a
        bounding box around the face and add a label.

        Args:
            frame_as_image (PIL Image): Original full image containing the faces
            detected_faces (array of Tuples): bounding box for each detected face
                that includes: top left corner position (x,y) as well as width and height

        Returns:
            A PIL Image with the original image overlayed with the bounding boxes and labels
            for each detected face
        '''

        # We need these local variables, so turn off Lint's complaint
        # pylint: disable=too-many-locals

        draw = ImageDraw.Draw(frame_as_image)
        for face in detected_faces:
            # get the top-left and lower-right coordinates of the bounding box for the face
            x_1, y_1, width, height = tuple(face)
            x_2 = x_1 + width
            y_2 = y_1 + height

            # generate a cropped image of the face with proper size to pass to the recognizer
            cropped_face = frame_as_image.crop((x_1, y_1, x_2, y_2))
            cropped_face = cropped_face.resize(self.embedding_image_dimensions)

            # This can be uncommented and used to see exactly what the cropped image looks like
            # cropped_face.save("cropped_face.jpg")

            # bounding box around face
            draw.rectangle(((x_1, y_1), (x_2, y_2)), outline='red')

            if not detect_only:
                # run the face recognizer on the image here
                name_for_face, process_time = self.face_recognizer.get_name_for_face(cropped_face)

                if name_for_face == "":
                    name_for_face = "Unknown"

                # label the face
                face_label = name_for_face + ' {:.3f}s'.format(process_time)
                face_label_width = self.face_label_font.getsize(face_label)
                face_label_start_x = x_1 + (x_2-x_1)/2 - face_label_width[0]/2
                draw.text((face_label_start_x, y_2 + 5), face_label, fill='red', font=self.face_label_font)

        # label the current FPS as well
        annotate_text = 'Processing time: {:.3f}s'.format(time.monotonic() - self.start_time_stamp)
        draw.text((175, 10), annotate_text, fill="red", font=self.fps_font)
