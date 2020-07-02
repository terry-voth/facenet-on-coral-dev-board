''' learn_faces.py

    Purpose: Generate embeddings for training faces
'''

import argparse

from numpy import asarray, savez_compressed, copy as copyarray

from face_detection_engine import FaceDetectionEngine, FaceDetectionMethodEnum
from face_embedding_engine import FaceEmbeddingEngine, FaceEmbeddingModelEnum
from utils import get_processed_training_data, \
                  TRAINING_FACE_IMAGES_OUTPUT_FILE, \
                  LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE, \
                  is_ubuntu_64, \
                  is_coral_dev_board

TRAINING_DATA_DIRECTORY = 'training_data/train/'
VALIDATION_DATA_DIRECTORY = 'training_data/val/'
SAVE_TRAINING_DATA = True
FACENET_EMBEDDING_MODEL_PATH = 'models/facenet_keras.h5' # trained with Celebrity imageset
FACENET_TFLITE_EMBEDDING_MODEL_PATH = 'models/facenet_keras_edgetpu.tflite'

# Ignore long lines as I'm using verbose variable names for easier understanding
# pylint: disable=line-too-long

def main(face_detector, face_embedder, face_embedder_model, skip_embeddings):
    ''' function main

    Main processing function - create face embeddings for all training images

    Args:
        face_detector (FaceDetectionEngine): Engine used to detect faces
        face_embedder (FaceEmbeddingEngine): Engine used to identify faces

    Returns:
        None
    '''

    # load training dataset
    training_images, training_labels = get_processed_training_data(TRAINING_DATA_DIRECTORY, face_detector, face_embedder_model)
    print("Total training data - images: {}, labels: {}".format(training_images.shape, training_labels.shape))

    # load validation dataset
    validation_images, validation_labels = get_processed_training_data(VALIDATION_DATA_DIRECTORY, face_detector, face_embedder_model)
    print("Total test data - images: {}, labels: {}".format(validation_images.shape, validation_labels.shape))

    if SAVE_TRAINING_DATA:
        # save arrays to one file in compressed format
        print("Saving image training data to: {}".format(TRAINING_FACE_IMAGES_OUTPUT_FILE))
        savez_compressed(TRAINING_FACE_IMAGES_OUTPUT_FILE, training_images, training_labels, validation_images, validation_labels)

    if not skip_embeddings:

        # convert each face in the training set to an embedding
        training_embeddings = list()
        for face_pixels in training_images:
            embedding = face_embedder.get_embedding(face_pixels)
            training_embeddings.append(copyarray(embedding))
        training_embeddings = asarray(training_embeddings)

        # convert each face in the validation set to an embedding
        validation_embeddings = list()
        for face_pixels in validation_images:
            embedding = face_embedder.get_embedding(face_pixels)
            validation_embeddings.append(copyarray(embedding))
        validation_embeddings = asarray(validation_embeddings)

        print("training embeddings shape: ", training_embeddings.shape)
        print("validation embeddings shape: ", validation_embeddings.shape)

        # save arrays to one file in compressed format
        print("Saving image training embeddings to: {}".format(LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE))
        savez_compressed(LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE, training_embeddings, training_labels, validation_embeddings, validation_labels)

parser = argparse.ArgumentParser()
parser.add_argument('--skip-embeddings', dest='skip_embeddings', action='store_true')
parser.set_defaults(skip_embeddings=False)
args = parser.parse_args()

if __name__ == '__main__':
    face_embedding_engine = None
    face_embedding_model = None
    if is_ubuntu_64:
        face_detection_engine = FaceDetectionEngine(FaceDetectionMethodEnum.MTCNN)
        face_embedding_model = FaceEmbeddingModelEnum.CELEBRITY_KERAS
        if not args.skip_embeddings:
            face_embedding_engine = FaceEmbeddingEngine(face_embedding_model)
    elif is_coral_dev_board:
        face_detection_engine = FaceDetectionEngine(FaceDetectionMethodEnum.SSD_MOBILENET_V2)
        face_embedding_model = FaceEmbeddingModelEnum.CELEBRITY_TFLITE
        if not args.skip_embeddings:
            face_embedding_engine = FaceEmbeddingEngine(face_embedding_model)
    else:
        raise Exception("Unsupported platform")

    if args.skip_embeddings:
        print("Skipping embeddings, only generating face images")

    main(face_detection_engine, face_embedding_engine, face_embedding_model, args.skip_embeddings)
 