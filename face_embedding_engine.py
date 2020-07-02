''' module face_embedding_engine.py

    Purpose: implement a face embedding engine
'''

from enum import Enum

from numpy import expand_dims, uint8 as np_uint8

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

# Pre-trained model with MS-Celeb-1M image set (https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
# Converted to Keras: https://github.com/nyoki-mtl/keras-facenet
FACE_EMBEDDING_CELEBRITY_KERAS_MODEL_PATH = 'models/facenet_keras.h5'

# Keras model converted to quantized tflite model (via convert_h5_to_tflite.py)
FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH = 'models/facenet_keras_edgetpu.tflite'

class FaceEmbeddingModelEnum(Enum):
    ''' enum FaceEmbeddingModelEnum

        Enumerates all models supported for identifying faces
    '''
    CELEBRITY_KERAS = 1  # Keras model pre-trained using  MS-Celeb-1M
    CELEBRITY_TFLITE = 2 # tflite version of CELEBRITY_KERAS

class FaceEmbeddingEngine:
    ''' class FaceEmbeddingEngine

        Purpose: generate embeddings for images of faces
    '''

    def __init__(self, embedding_model):
        ''' function constructor

        Constructor for FaceEmbeddingEngine

        Args:
        embedding_model (FaceEmbeddingModelEnum): The model to use for generating
                        embeddings for face images

        Returns:
            None
        '''

        # We only want to import these modules at run-time since
        # they will only be installed on certain platforms.
        # pylint: disable=import-outside-toplevel, import-error

        self.embedding_model = embedding_model
        self.required_image_shape = get_image_dimensions_for_embedding_model(embedding_model) + (3,) # need 3 arrays for RGB

        if self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_KERAS:
            print("Using Celebrity trained Keras model for face embeddings")
            from keras.models import load_model
            self.face_embedding_engine = load_model(FACE_EMBEDDING_CELEBRITY_KERAS_MODEL_PATH, compile=False)
        elif self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_TFLITE:
            print("Using Celebrity trained tflite model for face embeddings")
            from edgetpu.basic.basic_engine import BasicEngine
            self.face_embedding_engine = BasicEngine(FACE_EMBEDDING_CELEBRITY_TFLITE_MODEL_PATH)
            print("Embedding model input tensor shape: {}".format(self.face_embedding_engine.get_input_tensor_shape()))
            print("Embedding model input size: {}".format(self.face_embedding_engine.required_input_array_size()))
        else:
            raise Exception("Invalid embedding mode method: {}".format(embedding_model))

    def get_embedding_model(self):
        ''' function get_embedding_model

        Get the embedding model being used by this instance of the FaceEmbeddingEngine

        Args:
            None

        Returns:
            The FaceEmbeddingModelEnum being used by this instance of FaceEmbeddingEngine
        '''
        return self.embedding_model

    # get the face embedding for one face
    def get_embedding(self, face_pixels):
        ''' function get_embedding

        Generate an embedding for the given face

        Args:
            face_pixels (cv2 image): The image of the face to generate the
                            embedding for. The dimensions of the image must
                            match the dimensions required by the selected
                            embedding model.

        Returns:
            A numpy array with the embedding that was generated
        '''

        # Confirm we're using a proper sized image to generate the embedding with
        if face_pixels.shape != self.required_image_shape:
            raise Exception("Invalid shape: {} for embedding mode method: {}".format(face_pixels.shape, self.embedding_model))

        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        sample = expand_dims(face_pixels, axis=0)

        # get embedding
        if self.embedding_model == FaceEmbeddingModelEnum.CELEBRITY_KERAS:
            embeddings = self.face_embedding_engine.predict(sample)
            result = embeddings[0]
        else:
            sample = sample.flatten()
            # normalize values to between 0 and 255 (UINT)
            sample *= 255.0/sample.max()
            # convert to UNIT8
            sample = sample.astype(np_uint8)
            embeddings = self.face_embedding_engine.run_inference(sample)
            result = embeddings[1]

        return result

def get_image_dimensions_for_embedding_model(embedding_model):
    ''' function get_image_dimensions_for_embedding_model

    Get the required dimensions for images to use with the given embedding model

    Args:
        embedding_model (FaceEmbeddingModelEnum): The model being used for generating
                        embeddings for face images

    Returns:
        A tuple of the required width,height dimensions for images used by the given embedding model
    '''
    result = None
    if embedding_model in (FaceEmbeddingModelEnum.CELEBRITY_KERAS, FaceEmbeddingModelEnum.CELEBRITY_TFLITE):
        result = (160, 160)
    else:
        raise Exception("Invalid embedding model: {}".format(embedding_model))

    return result
