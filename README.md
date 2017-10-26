# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I used deep neural networks and convolutional neural networks (CNN) to clone driving behavior. The CNN was trained, validated and tested a model using Keras. The model output a steering angle to an autonomous vehicle.

I used a car simulator with steering capability around a track for data collection. This simulator was provided with the project. I used this image data and steering angles to train a neural network and then this model to drive the car autonomously around the track.

For a detailed project writeup please check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md).

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [Behavioral Cloning environment ](https://github.com/pmishra02138/CarND-Behavioral-Cloning-P3/blob/master/environment.yml)

The lab environment can be created with behavioral cloning environment.The following resources can be found in this github repository:

* createModel.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* writeup_report.md (a report writeup file)
* video.mp4 (a video recording of vehicle driving autonomously around the track)

The simulator can be downloaded from the udacity classroom material.

## Details About Files In This Directory

#### Create a model using Keras
To create a convolutional neural network using Keras from training image, run following command:

```sh
python createModel.py
```

This creates a model file, `model.h5`, that can be run by `drive.py`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

#### Run the trained model

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which the images seen by the agent is saved. If the directory already exists, it'll be overwritten. These images are used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video name is `run1.mp4`.
