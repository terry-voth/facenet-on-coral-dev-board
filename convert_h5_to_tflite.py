''' module face_detector.py

    Purpose: Take a Keras H5 model and quantize/convert it to a tflite model.
             Before running on an Edge TPU device (like the Coral dev board)
             the tflite model must first be compiled using the "edgetpu_compiler"
             utility. Note that, obviously, this can only be run on Ubuntu and
             not on the Coral dev board.

    Preparation: Generate the face images that you will be using to test the
                 face recognition with first and then run this utility. It will
                 use those images as representative input when converting the model.
'''

import tensorflow as tf
from numpy import load
from numpy import expand_dims

from utils import TRAINING_FACE_IMAGES_OUTPUT_FILE
from face_embedding_engine import FACE_EMBEDDING_CELEBRITY_KERAS_MODEL_PATH, \
                                  FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

# load the face dataset to use as representative data
data = load(TRAINING_FACE_IMAGES_OUTPUT_FILE)
training_images, training_labels, validation_images, validation_labels = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', training_images.shape, training_labels.shape, validation_images.shape, validation_labels.shape)

def representative_dataset_gen():
    ''' function representative_dataset_gen

    Feed representative data to the conversion process so it knows
    the range of typical values that the model will be processing

    Args:
        None

    Returns:
        Yields a sample input to inference with using the Celebrity
        trained Keras model
    '''
    for face_pixels in training_images:
        # cast int values to float
        float_face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = float_face_pixels.mean(), float_face_pixels.std()
        float_face_pixels = (float_face_pixels - mean) / std
        # transform face into a sample of one
        sample = expand_dims(float_face_pixels, axis=0)
        print("Sample shape: ", sample.shape)
        yield [sample]

# perform the quantized conversion
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(FACE_EMBEDDING_CELEBRITY_KERAS_MODEL_PATH)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quantized_model = converter.convert()

# save the new converted model
open(FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH, "wb").write(tflite_quantized_model)
