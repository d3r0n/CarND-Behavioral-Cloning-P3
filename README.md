
[//]: # (Image References)

[image1]: ./images/header.png "Car simulator"
[image2]: ./images/center.gif "Center driving"
[image2.1]: ./images/left.jpg "Left camera"
[image2.2]: ./images/center.jpg "Center camera"
[image2.3]: ./images/right.jpg "Right camera"
[image2.4]: ./images/water.gif "Lake"
[image3.1]: ./images/recover_bad.gif "Side recovering done wrong!"
[image3.2]: ./images/recover.gif "Side recovering"
[image4]: ./images/obama.jpg "Not good enough"
[image5]: ./images/crazy.gif "Crazy track"
[image6]: ./images/turn_right.jpg "Turn right"
[image7]: ./images/turn_left.jpg "Turn left"
[image8]: ./images/shift_one.jpg "Shift"
[image9]: ./images/shift.png "Shift examples"
[image10]: ./images/shear_one.jpg "Shear"
[image11]: ./images/shear.png "Shear examples"
[image12]: ./images/brightness.png "Brightness"
[image13]: ./images/all.png "All"
[image14]: ./images/result.png "What model sees"
[image15]: ./images/loss_1.png "Loss model 1"
[image16]: ./images/sharp_turn.gif "Sharp turn"
[image17]: ./images/kim.jpg "Not bad"
[image18]: ./images/distribution.png "Distribution without generation"
[image19]: ./images/distribution_generated.png "Distribution with generation"
[image20]: ./images/loss_2.png "Loss model 2"
[image21]: ./images/turn_fixed.gif "Turn fixed"

[image99]: ./images/model.png "Model Architecture"
# Behavioral Cloning

![alt text][image1]

>__TLDR;__ here is a link to my [model code.](https://github.com/d3r0n/sdcen-behavioral-cloning/blob/master/model.py)

---

### You're reading it! :rocket: Great!

In this project I have:
* Used driving simulator to collect data of good driving behaviour. Yeah, for the sake of the argument lets say 'good' behaviour. I am certainly not Michael Schumacher. :smile:
* Build a convolution neural network in Keras that predicts steering angles from collected images.
* Trained and validated the model with a training and validation set (4-1 proportion).
* Made a video! :movie_camera: of car successfully driving around track without leaving the road. :tada: :checkered_flag:

---

### Model Architecture and Training Strategy

#### 1. Data, data, data...

From experience I can say that the biggest win in prediction is not a model but good data. With a good data you can win even with simple regression. So lets start with describing a data preparation.

To capture good driving behaviour, I first recorded two laps using center lane driving.
Here is an example image of center lane driving:

![alt text][image2]

Udacity simulator is really nice one. It captures images from center of the hood 'camera' as well as from left and right side 'cameras'. Of course, I in my dataset I have used captured images from all of them. But I could not use them 'just like that'. Why? because the steering angle will not be correct for the sides. Have a look:

![alt text][image2.1]
![alt text][image2.2]
![alt text][image2.3]

If model used all them for learning with the same steering angle it would be certainly wrong! Side images are shifted and if consider them as middle of the car you have to adjust them. Therefore, left sample angle is corrected with +0.25 and the right sample with -0.25.

At this point, I have created some simple network to check out how well it performs with current data. Effect was not spectacular, but the model ware constantly lost when car reached the side of the road. The car just continued into the lake... The problem here was that it did not know how to recover back to center.

![alt text][image2.4]

So, I recorded the two extra laps with side recovering. I did not know how should recover look like. Since car is leaning to the side it should drive to middle right? so did this:

![alt text][image3.1]

Bad mistake. Now think how does this movement look like from the front camera?
Yeh, model sees that when driving straight I want to turn! into the woods... OK, that is how recover should look like:

![alt text][image3.2]

It definitely helped the model but it was still not enough of data.

![alt text][image4]

Then, I have realised something :bulb: What if I drive in opposite direction?
Boom! :collision: Completely new track data! Hint: it is easier to record side recovering for left side of the road when driving track clockwise.

So far, so good. But there must be a better way to get more data and not to repeat same laps. Something which could help model to generalize.

---

#### Data augmentation.

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting one can simply collect more data (if it is possible). If not, you have to think of reducing features or changing the model.

> __Keras generators are very powerful. They allow to create routines to randomly augment both training and validation sets and create even more data to fed our models.__

##### Transform!
First random transformation was flipping images and angles. So if I had turn right with angle X after flipping the image I had turn left with angle -X. Simple. Data x2.

![alt text][image6]
![alt text][image7]

Next was image shifting. I created a routine to randomly shift image and change steering angle. For instance if I shifted image left I reduce the angle by num_pixels_shifted x 0.004. If I did not do it, the data will get messed up. Same as with side camera shots. The downside is that you end up with one extra parameter to tune.

Here how shift look like:

![alt text][image8]

To make it better. I have used nearest pixels to cover shifted are. End effect:

![alt text][image9]

Next transformation was random spatial shear. Example describes it best:

![alt text][image10]

Also here, to make it look better I have used nearest pixels to cover transposed areas.

![alt text][image11]

Because some areas ware more shady than others I have added random brightness transformation. Have a look:

![alt text][image12]

Combined transformations together:

![alt text][image13]

My last data augmentation was to crop images from 160x320 to 106x320 in order to remove hood of the car and the sky.

![alt text][image14]

Great ! :rocket:

About the cropping. In Keras you can add extra layer called Cropping2D to your network. This way the operation will processed on GPU and not on CPU. In some cases when you have lot transformations learning my slow down. Not because of GPU but because of slow generator thread on CPU. You can use Cropping2D if you want to save CPU. But there is a better way. Multithreading! If make generator thread safe and adjust 'workers' count when calling model.fit_generator you will get rid of any data processing bottlenecks!. Here, have a look at my [data preparation code](https://github.com/d3r0n/sdcen-behavioral-cloning/blob/master/data.py) where you can find later solution.

---

#### 2. Network Design

After reading article on [NVIDIA tech blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) about their SDC I have decided to start with architecture similar to theirs.

So same as in NVIDIA, my model consists of 5 convolutional layers. Depths of the convolutions are between 32 and 128. All convolutional layers are followed by RELU to introduce nonlinearity. In my solution, I have added 2 max pooling layers in order to reduce computations for upper layers. Especially after flattening in fully connected layers.

> __I have added dropout to all layers to prevent model from [overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).__

 To normalize the input of the network I have added a BatchNormalization layer to normalize pixels from 0 - 255 space to -1 - 1.

5 Fully connected layers followed convolutional layers. After some initial experimentation I have decided to remove first of them. Additionally, I have used exponential relu as activation function. I hoped that leaky relu will make transition between angles smoother. I introduced dropouts and prepared learning strategy.

> __The model used an adam optimizer, so the learning rate did not need to be tuned manually.__

I found that it is better not to set too small starting learning rate for adam. So I went with 8e-4. Also, I have added early stopping in case model did not improve in 5 consecutive epochs. In general, early stopping helps overtraining model because it detects and stop training when validation loss go up. Additionally to pick best model I have added checkpoints every epoch and to increase visibility I have attached tensorboard logging.

If you are interested more in the architecture please have a look at the figure at the end of this document.

Then, I split my image and steering angle data into a training and validation set and made network learn.
This is how it went:

![alt text][image15]

Initially I have set 40 epochs. But as you can see early stopping worked and detected that after 17th epoch validation started to go up.

It was time to fire up the simulator!

![alt text][image16]

Not bad!

![alt text][image17]

Yeah, but not good enough either. Sharp sandy turns ware a problem and one of the wheels slightly left the track.

---

#### 3. Retraining.

Lets get back to data. To improve driving behaviour in above cases I recorded 4 more laps of sharp turns. Increasing number of samples to around 30k from 23k.

> __At the end, I had recorded 30,000 samples of total size 500 MB__

Here is distribution of the angles in the dataset.
(30,0000 samples)

![alt text][image18]

Now look here. It is distribution of the angles of the training samples as they are returned from batch generator.
(Generation of 60,000 samples)

![alt text][image19]

Nice, right? But, apart from data I needed to add one more layer. Should I start my training from the beginning?

> __Think twice before throwing away your previously trained model, you never know when it might be useful.__

So instead of throwing away my previous model (and money I paid Amazon...) I have used weights from already trained convolutional layers. Added new fully connected layer and randomized others. Voila!

Here is how the training went:

![alt text][image20]

And here is the final result:

![alt text][image21]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road! :tada: :checkered_flag:

---

#### Further improvements, aka "TODO"
Intentionally I did not show second track to the model to check how it generalize. After thoughtful analyse, I believe there is a room for a improvement :wink:

![alt text][image5]

In next iteration speed should be also included in the prediction.

#### Appendix 1
Here is a visualization of the architecture:

![alt text][image99]
