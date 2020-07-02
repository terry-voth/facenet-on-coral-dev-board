# README #

## Purpose ##

This intention of this project is to provide a practical and functional demonstration of facial recognition using TensorFlow on the Google Coral dev board

## Project Description ##

### Some Background ###

Recently I found myself with some spare time so I thought I'd spend some time taking a look at some Machine Learning. Specifically, facial recognition. A colleague mentioned the Google Coral dev board and I soon found myself with a brand new Google Coral dev board and Coral camera.

I've also only previously ever used Python for simple scripting and automation, so this was also a great opportunity to get into a little more advanced Python programming.

To begin with, I started with my Ubuntu laptop. There was a plethora of resources on the web to jump right in with face detection and face recognition so I was up and running quickly with a working demo on my laptop. I then tried to get the demo working on my new Coral dev board. That's where things got interesting.

After searching the net I was not able to find a single good example of face recognition on the Coral dev board. My intention was to be able to take pictures of my face using the Coral camera connected to the dev board and then be able to, using those pictures, recognize my face from a video steam using the same camera. I also wanted to publish the video stream from the dev board in a simple web server that would consist of the images captured from the camera along with any augmentation to those images on my part (i.e., boxes around detected faces and labels for any faces that are recognized). I could find no such demo anywhere, hence this project.

It appears that FaceNet is the current method for face recognition so that's the strategy I followed. This blog post was a great reference for detailing that strategy: [https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)

This blog post also had a pointer to the pre-trained Keras FaceNet model that I used. That model can be downloaded from [here](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)

### The Tricky Parts ###

Getting a live demo up and running on an Ubuntu laptop was pretty straight forward. Transferring that demo to the Coral dev board and utilise its TPU proved a little tricky.

#### Converting a Keras FaceNet model to tflite ####

In order to run the demo on the Coral dev board the Keras FaceNet model I used to generate the face embeddings needs to be 'quantized' and then compiled into a form that the TPU can use (e.g., integer based). There were a number of resources I found on the net that did this with other models or a different version of TF, but none of them worked for my FaceNet model. The [convert_h5_to_tflite.py](convert_h5_to_tflite.py) file is a culmination of the different aspects of these resources that I pulled together to get a working conversion.

#### Installing OpenCV on the Coral dev board ####

I used the OpenCV Python module to capture images from the Coral camera. I first tried installing a pre-built Python module for OpenCV but noticed that I couldn't capture full-size images from the Coral camera. Instead I had to install the source code and build it directly on the Coral dev board. Details are further down.

Also note that when installing the scikit-learn Python wheel package it will take about 2 3/4 hours to complete. The required build tools are installed when building the OpenCV module.

#### Using Proper Face Images ###

At first I captured images for face embeddings on my Ubuntu laptop then transferred them to the dev board to use for generating the face embeddings from the converted tflite model. The resulting facial recognition was very low quality.

On Ubuntu I was using the MTCNN Python module to detect and capture the face images to generate the embeddings with. MTCNN can't be installed on the dev board so I used the MobileNet V2 SSD model for face detection on the dev board. Mixing the two was a bad idea and was the cause of my poor results. So I added the ability to grab face images using the dev board's camera (using MobileNet V2 SSD) and then used those images for generating the face embeddings. This resulted in much better recognition.

Personally I prefer the MTCNN face captures as it seems to provide a tighter bounding box around the face which I think would create more pixels to use to generate the face embeddings. In the future I'd look at seeing if I could get MTCNN installed on the dev board.

## Supported Platforms ##

The demo supports the following environments:

* Ubuntu 18.04 - with laptop web camera
  * Uses MTCNN to detect faces
* Coral dev board - with Coral camera (and USB mouse to help capture training images)
  * Uses  MobileNet V2 SSD (Faces) to detect faces

The demo will automatically detect the platform and use the appropriate models for that platform, so no extra command line parameters are required.

## Set Up ##

### Set Up Coral Dev Board and Camera ###

Follow the Coral Dev board 'Get Started' guidelines to setup the dev board after first receiving it (or to reset it back to an initialized state). This should include installing TensorFlow Lite. The instructions are here: [https://coral.ai/docs/dev-board/get-started/](https://coral.ai/docs/dev-board/get-started/)

The dev board needs to be connected to the same network as your computer.

The camera also needs to be connected to the dev board. The instructions are here: [https://coral.ai/docs/dev-board/camera/](https://coral.ai/docs/dev-board/camera/)

You may want to add the following line to your ~/.bashrc file on the Coral dev board for convenience:

```shell
    export PATH=$PATH:/home/mendel/.local/bin:/sbin
```

### Install Dependencies ###

#### Building OpenCV ####

I downloaded and installed an OpenCV Python binary I was able to find and tried it but found that I could not read full sized images from the camera, therefore I installed the source code for OpenCV and built it directly on the Coral dev board.

NOTE: Building OpenCV natively on the Coral dev board takes a long time - about 8 hours!

I found instructions on how to build OpenCV on the Coral dev board on this great blog post: [https://krakensystems.co/blog/2020/doing-machine-vision-on-google-coral](https://krakensystems.co/blog/2020/doing-machine-vision-on-google-coral)

The Coral dev board will run out of memory if you try to build OpenCV (or, later on, scikit-learn) so you need to install a USB stick or micro SD card and setup a swap drive on it. The blog above included instructions but I add a few more details to make the swap drive permanent and lower the 'swappiness' of if so it isn't used aggresively.

Full detailed instructions on setting up the SD card with swapfile and building the OpenCV module can be found in [BUILD_OPENCV.md](BUILD_OPENCV.md)

#### Install Remaining Dependencies ####

These Python modules are also required:

```shell
    pip3 install flask pillow numpy evdev
```

Finally, we need the scikit-learn module. Note that this module will take about 2 3/4 hours to build/install:

```shell
    pip3 install scikit-learn
```

### Copy Source Code to Coral dev board ###

Use the 'mdt' command line tool to transfer the source files (including models, training images, etc) to the dev board. The Keras model (models/facenet_keras.h5) can be skipped as it's a larger file and can not be used by the Coral dev board TPU.

## Execute the demo ##

* General notes:
  * you need at least 1 training image and one validation image
  * you need at least 2 different people to 'train' with
  * Training images for Mathew McConaughey and Drake are provide to illustrate the required directory structure
* If running the demo on the Coral dev board, open a 'mdt shell' terminal
* Change to the facial-recognition/src directory
* For each face you want to test recognition on:
  * run this command (see more details on this command further down) to capture images of the face to recognize (you can experiment using only 2 or 3 images, or using more images, like 10 or 20):

    ```shell
    python3 server.py --capture
    ```

  * create two subdirectories under each of the following directories with the same directory name (equal to the name of the face):
    * ./training_data/train
    * ./training_data/val
  * copy about 3/4 of the captured images to the training subdirectory you just created
  * copy the remaining captured images to the validation subdirectory you just created
* Execute this command to generate face embeddings for the training faces:

  ```shell
  python3 learn_faces.py
  ```

* Find the IP address of the dev board by executing this command:

  ```shell
  /sbin/ifconfig
  ```

* Run the demo by executing this command:

  ```shell
  python3 server.py
  ```

* From your computer, browse to the web server running on the dev board: \<dev board IP address\>:5000

## Capturing test images ##

On the Coral dev board, before running the server program you must ensure that the mouse is associated with /dev/input/event2. Do this by:

* power off the Coral dev board and make sure no mouse is connected (it's assumed that the power cable and laptop are connect to 2 USB ports)
* power on the dev board and wait for it to completely start up
* connect a mouse to the USB connector beside the Ethernet connected

Execute this command to capture images from the camera that can be used in training:

```shell
python3 server.py --capture
```

Open a web browser to \<dev board IP address\>:5000 to start loading images from the camera and see the images being streamed.

On Ubuntu, put the opened window in focus and press the space bar to capture an image.

On the Coral dev board click the left mouse button to capture an image

All images captured will be stored to the current directory with the name "captured_image_x.jpg" where "x" is an increasing index number starting from 1.

Restarting the program and capturing images will overwrite any existing images.

It's recommended to use this method on the device you're going to run the face recognition demo on in order to capture the face images to 'train' with.

## Results ##

For face detection the total time required (on the Coral dev board) is approx. 35ms to 40ms which is good enough for about 24 FPS.

Total face recognition time (which includes the initial face detection) can take up to 500ms (for a single face), so not in the realm of real-time. However, using some software tricks (like caching the bounding box for each face and using that cache to compare in frames between the facial recognition frames) could feign that somewhat.
