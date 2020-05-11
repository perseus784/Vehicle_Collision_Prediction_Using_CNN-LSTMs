# A View NOT to Kill

## About
<img src="media/result1.gif" align="right" width="430" height="280"> 

The project combines both CNNs and LSTMs to predict whether a vehicle is in a collision course using a series of images moments before it happens. 

## Configuration
* Python 3.7
* TensorFlow 1.14
* Carla 0.9.5

## Data Collection 

<img align="right" src="media/dataset_structure.png" width="300" height="200">

## Model Architecture

<img align="right" src="media/final_Network_arch.png" width="350" height="680">


## Training
It is a decently large network with around 14 million parameters and it required a really good GPU for training or it ran into memory exhaust errors. I used a GTX 1080 Ti machine in which the training was done within 3 hours. 

<p align="center">
<img src="media/train_accuracy.png" width="400" height="400">
<img src="media/inception_valid.jpg" width="400" height="400">
</p> 


## Results
<img align="right" src="media/result2.gif" width="450" height="290">


*Full Video: https://www.youtube.com/watch?v=5E20U7b_4zQ*
