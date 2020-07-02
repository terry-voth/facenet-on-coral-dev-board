''' module server.py

    Purpose: implement a basic Flask web server publishing a
             video stream based on jpg images
'''

import time
import argparse
from flask import Flask, render_template, Response
from video_camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    ''' function index ...

    Args: None

    Returns:
        rendered index.html template
    '''
    return render_template('index.html')

def gen(camera):
    ''' function gen

    A Generator producing the jpg images that will be served as the frames for the video stream

    Args:
        camera (VideoCamera): Instance of a VideoCamera that will provide the jpg images

    Yields:
        individual video frames that are jpg encoded
    '''
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        else:
            # Sleep 10ms to avoid a really tight loop if there's no frame ready
            time.sleep(.01)

@app.route('/video_feed')
def video_feed():
    ''' function video_feed()

    Function that Flask will be call to retrieve individual frames for the video stream.

    Args: None

    returns:
        jpg frames produced by our video frame generator
    '''
    return Response(gen(VideoCamera(args.capture, args.detect_only)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Support option to capture images
parser = argparse.ArgumentParser()
parser.add_argument('--capture', dest='capture', action='store_true')
parser.set_defaults(capture=False)
# Support option to only detect images (not try to recognize faces)
parser.add_argument('--detect-only', dest='detect_only', action='store_true')
parser.set_defaults(detect_only=False)

args = parser.parse_args()

print("Path: ", app.instance_path)
if __name__ == '__main__':

    # start local web server
    app.run(host='0.0.0.0', port='5000', debug=True)
