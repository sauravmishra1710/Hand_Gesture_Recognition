# Hand_Gesture_Recognition


Gesture recognition is a topic in computer science and language technology with the goal of interpreting human gestures via mathematical algorithms. Gestures can originate from any bodily motion or state but commonly originate from the face or hand. Current focuses in the field include emotion recognition from face and hand gesture recognition. Users can use simple gestures to control or interact with devices without physically touching them. Many approaches have been made using cameras and computer vision algorithms to interpret sign language. However, the identification and recognition of posture, gait, proxemics, and human behaviors is also the subject of gesture recognition techniques. Gesture recognition can be seen as a way for computers to begin to understand human body language, thus building a richer bridge between machines and humans than primitive text user interfaces or even GUIs (graphical user interfaces), which still limit the majority of input to keyboard and mouse and interact naturally without any mechanical devices. Using the concept of gesture recognition, it is possible to point a finger at this point will move accordingly. 
(Ref -https://en.wikipedia.org/wiki/Gesture_recognition)


## Problem Statement

You want to develop a feature in the smart-TV that can recognize five different gestures performed by the user which will help users control the TV without using a remote.
Each gesture corresponds to a specific command:
    • Thumbs up: Increase the volume
    • Thumbs down: Decrease the volume
    • Left swipe: 'Jump' backwards 10 seconds
    • Right swipe: 'Jump' forward 10 seconds
    • Stop: Pause the movie
    
## Data

The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames (images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - like what the smart TV will use.

    1. Training
    2. Validation
    
Each training and validation set consists of several sub folders. Each subfolder consists of frames of one video (30 frames for a gesture). Also, two csv files which consist of the video metadata (i.e. path of the video and its label).
All images/frames of a video are of same dimension, there are two variants of the video image dimensions: 360x360 and 120x160

## Solution Approach

The following steps were followed in-order to classify the gestures correctly -

    1. Data Understanding.
    2. Data preprocessing and cleaning.
    3. Define the model architectures
    4. Train the models.
    5. Model performance comparison.
    6. Final model selection.

Let’s understand each phase one by one-

### Data Understanding

Data provided consists of the training and validation sets. Each set consists of several sub folders, each sub folder having 30 frames.
Input video frame dimensions:

    • 360x360x3
    • 120x160x3
    
### Preprocessing

Following two steps are performed in preprocessing:

    1. Resize: In the preprocessing step we have converted two different size of frames into one dimension of size 120x120. 
    We have images are of 2 different shape and the conv3D will throw error if the inputs in a batch have different 
    shapes. We will crop the image to (120, 120, 3), if the shape is (120, 160, 3)and we will resize to (120, 120, 3), 
    if the shape is (360, 360, 3)

    2. Normalization: Rescale pixel values from the range of 0-255 to the range 0-1 preferred for neural network 
    models. Image data is normalized by dividing pixel values by 255.
    
    
Along with the data normalization and resizing, we need to implement generator function. which will feed data to the model for each epoch. We need to implement generator function which can feed equal size batches of sequences and take care of remaining sequences.
For example, with training size of 23 sequences and batch size of 5 leads to 5 batches as below -

    Batch-1: 5 sequences
    Batch-2: 5 sequences
    Batch-3: 5 sequences
    Batch-4: 5 sequences
    Batch-5: 3 sequences

### Model Architecture
There are 3 different deep learning architectures which can be used to build model for gesture recognition. Let’s see those architectures below.

#### CNN + RNN

This is the standard architecture for processing videos. In this architecture, video frames are passed through a CNN layer which extracts features from the images and then these feature vectors are fed to an RNN network to simulate sequence behavior of the video. Output of RNN is regular SoftMax function
      1. We can use transfer learning in 2D CNN layer instead of training own network.
      2. LSTM or GRU can be used in RNN
      
#### 3D Convolution Network or Conv3D

3D convolutions are a natural extension to the 2D convolutions. Just like in 2D conv, we move the filter in two directions (x and y), in 3D conv, the filter is moved in three directions (x, y and z).

In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels.

Hence, deriving the analogy from 2-D convolutions where a 2-D kernel/filter (a square filter) is represented as (fxf)xc where f is filter size and c is the number of channels, a 3-D kernel/filter (a 'cubic' filter) is represented as (fxfxf)xc (here c = 3 since the input images have three channels).

This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.

#### Transfer Learning

We will use some pre-trained models and try to use their knowledge to classify our cases correctly. These pre-trained models which are trained on millions of images may prove vital in solving our problem efficiently. For transfer learning -
We have done our experiments with the following pre-trained models -

    1. MobileNet
    2. InceptionResNetV2
    3. EfficientNetB0

We experimented several such models in the 2D CNN layer.
We experimented with GRU units as well as LSTM units.
We also experimented transfer learning by setting the layers of the pre-trained models set to –
    1. Trainable - False
    2. Trainable – True
    
Many models were tested based on the above architectures. These models were created taking into consideration the following model parameters while performing a series of experiments throughout the course of the project –

    • Depth of the model - The number of layers a model could contain.
    • Batch Normalization for the hidden layers.
    • Dropout layers to de-activate some of the neurons randomly.
    • Dense Layer architecture.
    • The Optimizer functions.
    • Adding the L2 regularization to some models.
    • Time Distributed Layers – LSTM’s and GRU’s.
    • Transfer Learning.
    
### Model Training History

One of the default callbacks that is registered when training all deep learning models is the History callback. It records training metrics for each epoch. This includes the loss and the accuracy (for classification problems) as well as the loss and accuracy for the validation dataset, if one is set.

The history object is returned from calls to the fit() or the fit_generator() function used to train the model. Metrics are stored in a dictionary in the history member of the object returned. For a classification problem the history callback returns - 
**['acc', 'loss', 'val_acc', 'val_loss']**

We have made use of the history callback to access the training history for our models and plot the loss, val_loss and accuracy, val_accuracy against the number of epochs.

The plots give us insights to -
    • It’s speed of convergence over epochs (slope).
    • Whether the model may have already converged (plateau of the line).
    • Whether the mode may be over-learning the training data (inflection for validation line).
    
(Ref - https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
