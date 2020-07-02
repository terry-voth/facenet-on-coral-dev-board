''' module utils.py

    Purpose: provide common functions/data for use by other modules
'''

from os import listdir
from os.path import isdir
import platform

from numpy import asarray, array as np_array
from PIL import Image

# Stores actual training image pixels for faces (images are clipped from training image
# and resized to the input size of the embedding we want to use)
TRAINING_FACE_IMAGES_OUTPUT_FILE = 'training_data/training-data-faces-dataset.npz'

# Stores embeddings for faces we want to identify
LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE = 'training_data/trained-faces-embeddings.npz'

# Set this to true to print out performance times for detecting/recognizing
PRINT_PERFORMANCE_INFO = False

def load_faces(directory, face_detection_engine, embedding_model):
    ''' function load_faces

    Load images and extract a single face for all files in a directory

    Args:
        directory (string): The directory to load the image files from
        face_detection_engine (FaceDetectionEngine): The engine to use for detecting faces
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A list of numpy arrays for faces detected in the image files in the given directory
    '''

    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # load image from file
        image = Image.open(path)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to numpy array
        rgb_array = np_array(image)

        # get face
        face = face_detection_engine.extract_face(rgb_array, embedding_model)
        # store
        if len(face) > 1:
            faces.append(face)
        else:
            print("Face not detected in file: {}".format(path))
    return faces

def get_processed_training_data(directory, face_detection_engine, embedding_model):
    ''' function get_processed_training_data

    Load a dataset that contains one subdir for each class that in turn contains images

    Args:
        directory (string): The directory to load the image files from
        face_detection_engine (FaceDetectionEngine): Detector being used to detect faces
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A Tuple of two numpy arrays:
            - An array of arrays:
                for each subdirectory
                    an array of faces detected in the image files
            - An array of arrays:
                for each subdirectory
                    an array of labels for the faces detected in the image files
    '''

    face_images, face_labels = list(), list()
    # enumerate folders, one per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that also might be in the directory
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path, face_detection_engine, embedding_model)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        face_images.extend(faces)
        face_labels.extend(labels)
    return asarray(face_images), asarray(face_labels)

# for Ubuntu we get something like: Linux-5.3.0-59-generic-x86_64-with-Ubuntu-18.04-bionic
# for Coral dev board we get something like: Linux-4.14.98-imx-aarch64-with-Mendel-mendel-day-day
platform = platform.platform()
print("Running on platform: {}".format(platform))

# publish flags indicating what platform we're currently running on
UBUNTU_FINGER_PRINT = "x86_64-with-Ubuntu"
CORAL_DEV_BOARD_FINGER_PRINT = "aarch64-with-Mendel"
is_ubuntu_64 = UBUNTU_FINGER_PRINT in platform
is_coral_dev_board = CORAL_DEV_BOARD_FINGER_PRINT in platform
