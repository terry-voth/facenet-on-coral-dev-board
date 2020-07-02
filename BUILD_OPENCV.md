# Build OpenCV 4.1.1 #

Unfortunately, as of the writing of this, OpenCV doesn't have a pre-build binary for Arm devices so we need to build it. From [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/):

```comment
"...the wheel (especially manylinux) format does not currently support properly ARM architecture so there are no packages for ARM based platforms in PyPI".
```

So instead we download the OpenCV source code and build it directly on the Coral dev board.

These instructions were shamelessly lifted from this great blog post with a couple of very minor changes: [https://krakensystems.co/blog/2020/doing-machine-vision-on-google-coral](https://krakensystems.co/blog/2020/doing-machine-vision-on-google-coral)

## Install a Micro SD card and setup a swap file on it ##

* power down board ('sudo shutdown now' from mdt shell as opposed to just powering it off without shutting down properly)
* install micro SD card on board (16GB minimum)
* boot up board
* in an 'mdt shell' terminal:
  * format and mount new micro SD Card:
    * if any partitions already exist on the SD Card, delete them first or skip to formatting the existing partition further down
    * gets device id for micro SD card (should be "/dev/mmcblk1")

        ```shell
        sudo fdisk -l
        ```

    * create new partition

        ```shell
        sudo fdisk /dev/mmcblk1
        Command (m for help): n
        ```

    * make this a primary partition, leave default values for first and last sector

        ```shell
        Command (m for help): p
        ```

    * Write it (by default fdisk will make this a ext4 partition)

        ```shell
        Command (m for help): w
        ```

    * format card

        ```shell
        sudo mkfs.ext4 /dev/mmcblk1p1
        ```

    * mount the card

        ```shell
        sudo mkdir /media/sdcard
        sudo mount /dev/mmcblk1p1 /media/sdcard
        ```

    * set proper permissions

        ```shell
        sudo chown -R mendel:mendel /media/sdcard
        ```

    * make the mount permanent (optional if you only want to use the micro SD card for this session, but if you reboot you have to manually remount again)

        ```shell
        sudo su
        echo "/dev/mmcblk1p1 /media/sdcard ext4 defaults 0 1" >> /etc/fstab
        exit
        ```

## Create swap file ##

* in an 'mdt shell' terminal:
  * create the swapfile

    ```shell
    sudo fallocate -l 2G /media/sdcard/swapfile
    sudo chmod 600 /media/sdcard/swapfile
    sudo mkswap /media/sdcard/swapfile
    sudo swapon /media/sdcard/swapfile
    ```

  * verify swapfile is setup properly

    ```shell
    sudo swapon --show
    ```

  * set a low 'swappiness' so the system doesn't use it aggressively

    ```shell
    sudo su
    echo "vm.swappiness=10" > /etc/sysctl.conf
    exit
    ```

  * make the swapfile peramanent (optional if you don't want it permanent)

    ```shell
    sudo su
    echo "/media/sdcard/swapfile none swap sw 0 0" >> /etc/fstab
    exit
    ```

## Build and Install OpenCV 4.1.1 ##

* in an 'mdt shell' terminal:
  * install build dependencies:

    ```shell
    sudo apt-get update
    sudo apt-get install build-essential cmake unzip pkg-config \
        libjpeg-dev libpng-dev libtiff-dev git libavcodec-dev \
        libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev \
        libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev
    ```

  * download OpenCV source code:

    ```shell
    cd /media/sdcard
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.1.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.1.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
    mv opencv-4.1.1 opencv
    mv opencv_contrib-4.1.1 opencv_contrib
    ```

  * download and apply patch to correct build issue:

    ```shell
    cd opencv/cmake
    git clone https://gist.github.com/pjalusic/7feb9cd722a437c0f6eea34622ca44c8.git
    cp 7feb9cd722a437c0f6eea34622ca44c8/OpenCVFindLibsPerf.diff .
    git apply OpenCVFindLibsPerf.diff
    ```

  * use a Python virtual environment to build in:

    ```shell
    cd ~
    sudo pip3 install virtualenv virtualenvwrapper
    sudo rm -rf ~/.cache/pip
    ```

  * add the following lines to ~/.bashrc:

    ```shell
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source /usr/local/bin/virtualenvwrapper.sh
    ```

  * apply the changes made to ~/.bashrc:

    ```shell
    source ~/.bashrc
    ```

  * begin working in the new Pyton virtual environment:

    ```shell
    mkvirtualenv cv -p python3
    workon cv
    ```

  * apparently this helps correct a potential build issue:

    ```shell
    pip3 install numpy

    ```

  * prepare to build OpenCV:

    ```shell
    cd /media/sdcard/opencv
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/media/sdcard/opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
        -D ENABLE_FAST_MATH=1 \
        -D ENABLE_NEON=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_V4L=ON \
        -D BUILD_EXAMPLES=ON ..
    ```

  * build OpenCV (this takes about 8 hours):

    ```shell
    make -j$(nproc)
    ```

  * install the newly built artifacts:

    ```shell
    sudo make install
    sudo ldconfig
    cd /usr/local/lib/python3.7/site-packages/cv2/python-3.7
    sudo mv cv2.cpython-37m-aarch64-linux-gnu.so cv2.so
    cd ~/.virtualenvs/cv/lib/python3.7/site-packages/
    ln -s /usr/local/lib/python3.7/site-packages/cv2/python-3.7/cv2.so cv2.so
    cp -ar /usr/local/lib/python3.7/site-packages/cv2 /usr/local/lib/python3.7/dist-packages

    ```
