''' module face_recognizer.py

    Purpose: identify a face
'''
import time
from enum import Enum

from numpy import load, expand_dims, asarray, dot, transpose, sqrt, linalg, array as np_array

from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

from utils import LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE, \
                  PRINT_PERFORMANCE_INFO, \
                  is_ubuntu_64, \
                  is_coral_dev_board

from face_embedding_engine import FaceEmbeddingEngine

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

class MatchDistanceCalculationMethodEnum(Enum):
    ''' enum MatchDistanceCalculationMethodEnum

        Enumerates all methods supported for calculating the distance
        between the embedding of an unknown face and the embedding(s)
        of a known face.
    '''
    COSINE_MEAN = 1            # Calculate mean of 'matched' trained embeddings and then measure angular distance from that to the face embedding
    LINEAR_NORMALIZED_MEAN = 2 # Calculate mean of 'matched' trained embeddings and then measure linear distance from that to the face embedding
    MEAN_LINEAR_NORMALIZED = 3 # Measure linear distance between face embedding and each 'matched' trained embedding then calculate mean of that

MATCH_CALCULATION_METHOD = MatchDistanceCalculationMethodEnum.COSINE_MEAN
IDENTIFCATION_PROBABILITY_THRESHOLD = 80 # percent

class FaceRecognizer():
    ''' class FaceRecognizer

        Purpose: identify images of faces
    '''

    def __init__(self, embedding_model):
        ''' function constructor

        Constructor for FaceRecognizer

        Args:
            embedding_model (FaceEmbeddingModelEnum): The model to use for generating
                            embeddings for face images

        Returns:
            None
        '''

        # Observed distances differed between Ubuntu and Coral dev board
        if MATCH_CALCULATION_METHOD == MatchDistanceCalculationMethodEnum.COSINE_MEAN:
            if is_ubuntu_64:
                self.matched_distance_threshold = .6
            elif is_coral_dev_board:
                self.matched_distance_threshold = .4
            else:
                raise Exception("Unsupported platform")
        elif MATCH_CALCULATION_METHOD == MatchDistanceCalculationMethodEnum.LINEAR_NORMALIZED_MEAN:
            if is_ubuntu_64:
                self.matched_distance_threshold = 12
            elif is_coral_dev_board:
                self.matched_distance_threshold = 8
            else:
                raise Exception("Unsupported platform")

        # load face embeddings for 'learned' faces
        data = load(LEARNED_FACE_EMBEDDINGS_OUTPUT_FILE)
        training_embeddings, training_labels, validation_embeddings, validation_labels = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        # group embeddings by label for future comparison
        self.trained_embedding_lists = {}
        trained_labels = []
        for index, label in enumerate(training_labels):
            if label not in self.trained_embedding_lists:
                self.trained_embedding_lists[label] = []
                trained_labels.append(label)
            self.trained_embedding_lists[label].append(training_embeddings[index])

        # calculate mean value of all embeddings for each label
        self.trained_embedding_mean_values = {}
        for label in trained_labels:
            mean_value_for_embeddings = np_array(self.trained_embedding_lists[label]).mean(axis=0)
            self.trained_embedding_mean_values[label] = mean_value_for_embeddings


        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        training_embeddings = in_encoder.transform(training_embeddings)
        validation_embeddings = in_encoder.transform(validation_embeddings)

        # label encode targets
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(training_labels)
        training_labels = self.out_encoder.transform(training_labels)
        validation_labels = self.out_encoder.transform(validation_labels)

        # fit classifying model
        self.classifying_model = SVC(kernel='linear', probability=True)
        self.classifying_model.fit(training_embeddings, training_labels)

        # load the FaceNet model to generate face embeddings with
        self.embedding_engine = FaceEmbeddingEngine(embedding_model)

    def get_name_for_face(self, face_image):
        ''' function get_name_for_face

        Given an image of a face, generate an embedding for it and
        try to identify it. Use the SVC model to identify the most
        likely candidate then compare the embedding to the mean
        value of the candidate's embeddings by determining the 'distance'
        between the two. If the distance is within a certain threshold
        then the candidate is considered a match.

        Args:
            face_model (PIL Image): The image of the face to try to
                            identify. The dimensions of the image must
                            match the dimensions required by the selected
                            embedding model.

        Returns:
            If a match is found, return the string name of the match,
            otherwise return an empty string
        '''

        start_time = time.monotonic()

        # get an embedding for the face image
        face_embedding = self.embedding_engine.get_embedding(asarray(face_image))
        generate_embedding_time = time.monotonic() - start_time
        if PRINT_PERFORMANCE_INFO:
            print("Generate embedding time: {:.3f}s".format(generate_embedding_time))

        start_time = time.monotonic()

        # run the embedding through classifier to see if we
        # can identify the face from the embedding
        sample = expand_dims(face_embedding, axis=0)
        yhat_class = self.classifying_model.predict(sample)
        yhat_prob = self.classifying_model.predict_proba(sample)

        # get name (AKA the class 'label')
        class_index = yhat_class[0] # only care about the top match
        class_probability = yhat_prob[0, class_index] * 100
        class_label = self.out_encoder.inverse_transform(yhat_class)[0]

        # The SVC model returns match probababilities to all 'known' faces
        # so it always returns results (and the total probability always
        # equals 100%). So take the top 'match' and calculate the 'distance'
        # to the face embedding we're examining. If the distance is within
        # a certain threshold then the match is considered a true match.
        name_for_face = ""
        if class_probability > IDENTIFCATION_PROBABILITY_THRESHOLD:
            if MATCH_CALCULATION_METHOD == MatchDistanceCalculationMethodEnum.COSINE_MEAN:
                matched_embedding = self.trained_embedding_mean_values[class_label]
                matched_distance = find_cosine_distance(matched_embedding, face_embedding)
            elif MATCH_CALCULATION_METHOD == MatchDistanceCalculationMethodEnum.LINEAR_NORMALIZED_MEAN:
                matched_embeddings = self.trained_embedding_lists[class_label][0]
                matched_distance = linalg.norm(matched_embeddings - face_embedding).mean()
            elif MATCH_CALCULATION_METHOD == MatchDistanceCalculationMethodEnum.MEAN_LINEAR_NORMALIZED:
                linear_distances = linalg.norm(np_array(self.trained_embedding_lists[class_label]) - face_embedding, axis=1)
                matched_distance = linear_distances.mean()

            print("distance from {} ({:.1f}% SVC match) = {:.1f}".format(class_label, class_probability, matched_distance))
            if matched_distance <= self.matched_distance_threshold:
                name_for_face = class_label+" ("+'{:d}%'.format(int(class_probability))+")"

        classify_embedding_time = time.monotonic() - start_time
        if PRINT_PERFORMANCE_INFO:
            print("Classify embedding: {:.3f}s".format(classify_embedding_time))

        return name_for_face, (generate_embedding_time + classify_embedding_time)

def find_cosine_distance(face_embedding_1, face_embedding_2):
    ''' function find_cosine_distance

    Given two face embeddings, calculate the cosine distance
    between the two

    Args:
        face_embedding_1 (embedding): embedding for face 1
        face_embedding_2 (embedding): embedding for face 2

    Returns:
        Cosine distance between the two face embeddings
    '''
    x_value = dot(transpose(face_embedding_1), face_embedding_2)
    y_value = dot(transpose(face_embedding_1), face_embedding_1)
    z_value = dot(transpose(face_embedding_2), face_embedding_2)
    return 1 - (x_value / (sqrt(y_value) * sqrt(z_value)))
