''' module video_camera.py

    Purpose: Implements a simple video camera image feeder.
             Read frames from the system video camera (/dev/video0).
             For each frame, pass it to the face detector to augment
             the image with any faces that are detected (and identified).

             Frames are read from the video camera in a separate thread
             in order to decouple the reading of frames from the processing
             and publishing of those frames. Reading a frame from the video
             camera is blocking so moving this to its own thread improves
             overall performance.
'''

import select
import threading
import time

# import the necessary packages
from cv2 import VideoCapture, \
                CAP_PROP_FRAME_WIDTH, \
                CAP_PROP_FRAME_HEIGHT, \
                CAP_PROP_FPS, \
                WINDOW_NORMAL, \
                COLOR_BGR2RGB, \
                COLOR_RGB2BGR, \
                namedWindow, \
                cvtColor, \
                imencode, \
                imwrite, \
                waitKey

from numpy import array as np_array
from evdev import InputDevice, ecodes

from face_detector import FaceDetector
from face_detection_engine import FaceDetectionMethodEnum
from face_embedding_engine import FaceEmbeddingModelEnum
from utils import PRINT_PERFORMANCE_INFO, is_ubuntu_64, is_coral_dev_board

# We use descriptive variable and function names so
# disable the pylint warning for long lines
# pylint: disable=line-too-long

# Reduce image size to increase performance
CAMERA_RESOLUTION = (1280, 720)

class VideoCamera():
    ''' class VideoCamera

        Purpose: implement the functionality of the video camera
    '''

    # We need these class members, so turn off Lint's complaint
    # pylint: disable=too-many-instance-attributes

    def __init__(self, capture_images=False, detect_only=False):
        ''' function constructor

        Constructor for VideoCamera

        Args:
            capture_images (bool): Runs the VideoCamera in 'capture' mode
                            where images are not processed through the ML
                            but rather allows the user to capture screenshot
                            images from the video stream and save them to disk.

        Returns:
            None
        '''

        # load camera
        print("Camera images being processed at resolution: {}".format(CAMERA_RESOLUTION))
        self.video = VideoCapture(0)
        self.video.set(CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        self.video.set(CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        print("Camera FPS set at {:4.1f}".format(self.video.get(CAP_PROP_FPS)))

        self.capture_images_mode = capture_images
        self.detect_only = detect_only

        if not self.capture_images_mode:
            # load face detection engine
            if is_ubuntu_64:
                face_detection_method = FaceDetectionMethodEnum.MTCNN
                embedding_model = FaceEmbeddingModelEnum.CELEBRITY_KERAS
            elif is_coral_dev_board:
                face_detection_method = FaceDetectionMethodEnum.SSD_MOBILENET_V2
                embedding_model = FaceEmbeddingModelEnum.CELEBRITY_TFLITE
            else:
                raise Exception("Unsupported platform")

            self.face_detector = FaceDetector(face_detection_method, embedding_model)
        else:
            print("Starting up in image capture mode")
            if is_ubuntu_64:
                # required in order to be able to get key presses
                namedWindow('frame', WINDOW_NORMAL)
            elif is_coral_dev_board:
                # On Coral dev board, use mouse plugged into USB port beside ethernet connector
                self.key_select = select.poll()
                self.input_dev = InputDevice('/dev/input/event2')
                self.key_select.register(self.input_dev, select.POLLIN)
            else:
                raise Exception("Unsupported platform")
            self.captured_image_count = 0

        # initialize shared variables and the protecting semaphore lock
        self.frame_as_rgb_array = None # stores the last frame read
        self.status = False
        self.closing = False # flag to indicate when we're shutting down
        self.lock_frame = threading.Lock()
        # kick off the frame reading thread
        self.frame_reading_thread = threading.Thread(target=self.read_frames, args=(self.lock_frame,))
        self.frame_reading_thread.start()

        # initialize time for calculating FPS
        self.last_time_stamp = time.monotonic()

    def __del__(self):
        ''' function destructor

        Destructor for VideoCamera - release the camera

        Args:
            None

        Returns:
            None
        '''

        # shutdown the frame reading thread
        self.closing = True
        # wait for the thread to finish
        self.frame_reading_thread.join()
        # release the camera
        self.video.release()

    def read_frames(self, lock):
        '''thread read_frames

        Thread to just continually read frames from video stream. When
        the class member 'self.closing' is set to True, thread will
        automatically terminate.

        Each time a video frame is read from the camera it is saved to
        the 'self.frame' class member. Any frame that is already stored
        in 'self.frame' will be overwritten.

        Args:
            lock (threading.Lock) - mutex semaphore to protect shared data
                                    ('self.frame' and 'self.status')

        Returns:
            None
        '''

        while not self.closing:
            if self.video.isOpened():
                # extract a frame - note that this frame is in numpy.ndarray format (BGR)
                status, frame = self.video.read()
                if status:
                    lock.acquire()
                    self.frame_as_rgb_array = frame
                    self.status = status
                    lock.release()

    def get_frame(self):
        ''' function get_frame

        This function retrieves the latest frame from 'self.frame' class
        member.

        If in 'capture' mode, it checks for user input to see if the frame
        should be written to disk. No changes are made to the frame.

        If not in 'capture' mode, the frame is processed by the ML to detect
        any faces in the frame and, if detected, try to identfy them.

        The ML will update the frame if any faces are detected and/or identified.

        Args:
            None

        Returns:
            If a video frame exists, it is returned as a jpg encoded cv2 image,
            otherwise it returns None.
        '''

        self.lock_frame.acquire()

        if not self.status:
            self.lock_frame.release()
            return None

        new_frame = None
        if not self.capture_images_mode:
            if PRINT_PERFORMANCE_INFO:
                print("============================================")
                start_time = time.monotonic()

            # convert frame to RGB which the face detector requires
            rgb_array = cvtColor(self.frame_as_rgb_array, COLOR_BGR2RGB)

            # clear shared data and release lock ASAP
            self.frame_as_rgb_array = None
            self.status = False
            self.lock_frame.release()
            if PRINT_PERFORMANCE_INFO:
                print("Convert to RGB array time: {:.3f}s".format(time.monotonic() - start_time))
                start_time = time.monotonic()


            # identify any faces in the picture
            frame_as_image = self.face_detector.identify_faces_in_frame(rgb_array, self.detect_only)
            if PRINT_PERFORMANCE_INFO:
                print("    Total face processing: {:.3f}s".format(time.monotonic() - start_time))
                start_time = time.monotonic()

            # convert PIL Image back to streamable frame format (array of pixels)
            rgb_array = np_array(frame_as_image)
            # Convert RGB to BGR
            frame = cvtColor(rgb_array, COLOR_RGB2BGR)

            ret_val, jpeg_image = imencode('.jpg', frame)
            if ret_val:
                new_frame = jpeg_image.tobytes()

            if PRINT_PERFORMANCE_INFO:
                print("Convert to jpg time: {:.3f}s".format(time.monotonic() - start_time))
        else:
            # save the current image
            self.capture_image_locked()

            ret_val, jpeg_image = imencode('.jpg', self.frame_as_rgb_array)
            if ret_val:
                new_frame = jpeg_image.tobytes()

            # clear shared data and release lock
            self.frame_as_rgb_array = None
            self.status = False
            self.lock_frame.release()

        return new_frame

    def capture_image_locked(self):
        ''' function capture_image_with_lock

        Save the current video frame to file

        The semaphore lock should be acquired before calling this function

        Args:
            None

        Returns:
            None
        '''

        capture_image = False

        # detect input from user
        # On Ubuntu laptop, look for keyboard spacebar being pressed.
        # On Coral dev board, connect mouse to USB port next to network
        # connector and look for left mouse press.
        if is_ubuntu_64:
            # get key from keyboard
            key = waitKey(1)
            if key & 0xFF == ord(' '):
                capture_image = True
        elif is_coral_dev_board:
            # On Coral dev board, get mouse click
            events = self.key_select.poll(1)
            if events:
                for event in self.input_dev.read():
                    if event.type == ecodes.EV_KEY and \
                            event.code == ecodes.BTN_LEFT and event.value == 1:
                        capture_image = True
        else:
            raise Exception("Unsupported platform")

        if capture_image:
            self.captured_image_count += 1
            output_filename = "captured_image_"+str(self.captured_image_count)+".jpg"
            imwrite(output_filename, self.frame_as_rgb_array)
            print("Captured image: "+output_filename)
