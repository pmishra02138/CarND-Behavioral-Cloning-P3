
## Behavioral Cloning Project

The goals/ steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* creatModel.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
center_2017_10_20_06_46_15_085

#### 3. Submission code is usable and readable

The `creaetModel.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An [NVIDIA architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) has been employed

NVIDIA model is implemented in `creatModel.py` from lines 47-57. Various layers of this model are as follows:


| Layer         		|     Paramters	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		|  RGB image  (160, 320, 3) 							|  
| Normalization layer         		|  Input/255.0 - 0.5 							|  
| Cropping         		|  RGB image  (70, 25, 3) 							|  
| Convolution Layer 1      	| 5x5, subsample 2x2, activation=RELU|
| Convolution Layer 2      	| 5x5, subsample 2x2, activation=RELU|
| Convolution Layer 3      	| 5x5, subsample 2x2, activation=RELU|
| Convolution Layer 4      	| 3x3, activation=RELU|
| Convolution Layer 5      	| 3x3, activation=RELU|                
| Flatten	      	|  |   
| Fully connected	1	| 100        		|
| Fully connected	2	| 50        		|
| Fully connected	3	| 10        		|
| Fully connected	4	| 1        		|  

#### 2. Attempts to reduce overfitting in the model

I employed a few simple strategies to reduce overfitting. Training model on too many iterations could make it very data specific and introduce overfitting. To avoid this, I used only 3 epochs to train the model. The other strategy, I employed was to randomly split underlying data into 80-20 ratio. 80% for training and 20% for validation (code line 61). Finally, the model was tested by running simulator in autonomous mode and ensuring that the vehicle could stay on the track. I want to to mention that I only used one round of data for all simulation. In future, it will be interesting to try dropout or other strategies to reduce overfitting.  

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (`creatModel.py` line 60).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet (not included in final submission). I started with LeNet model as it worked very well for traffic sign classifier problem in previous project. Both traffic sign classifier and behavioral cloning are centered around computer vision and are *somewhat* similar problems. Unfortunately, LeNet model didn't produce good results and the car couldn't complete a full track in autonomous mode.  

Then I tried, NVIDIA model and it gave very good result. The car was successfully able to run full track in autonomous mode.

I used central, left and right images for this exercise. I supplemented the data with augmented images. Image augmentation was done by flipping original images.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set(20%). I trained the model for 3 epochs only. These two simple steps worked very well and avoided from overfitting the model to training data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I applied a correction factor of 0.2 from left and right images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`createModel.py` lines 47-57) consisted of a convolution neural network with the following layers and layer sizes as described in table above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track using center lane driving. I used all three left, right and center images for building model. One such sample image set is shown below.


| ![LeftImage](images/left_2017_10_20_06_46_15_085.jpg)| ![ CenterImage](images/center_2017_10_20_06_46_15_085.jpg) | ![ RightImage](images/right_2017_10_20_06_46_15_085.jpg) |
|:---:|:---:|:---:|
| Left image | Central image | Right image |

To augment the data sat, I flipped images and angles thinking that this would help balancing the training data. One such image is shown below.

| ![CenterImage1](images/center_2017_10_20_06_46_15_085.jpg)| ![ FlippedCenterImage](images/center_2017_10_20_06_46_15_085_flipped.jpg) |
|:---:|:---:|
| Central image | Flipped image |

After the collection process, I had 12,528 (4176x3) number of data points. After augmentation, there were a total of 25,056 (12,528x2) data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
