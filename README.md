# A View NOT to Kill

## About
<img src="media/out.gif" align="right" width="400" height="250">
The aim of the project is to predict whether a vehicle is in a collision course using a series of images **moments before it happens**. The project combines both CNNs and LSTMs to get time series prediction using image data. Various phases and challenges faced during this project is discussed.

## Configuration
* Python 3.7
* TensorFlow 1.14
* Carla 0.9.5

## Data Collection
<p align="right">
<img src="media/dataset_structure.png" width="300" height="200">
</p> 

<p align="right">
<img src="media/imgseq.png" width="400" height="700">
</p> 

## Model Architecture
<p align="center">
<img src="media/CNN-LSTM.png" width="800" height="400">
</p> 

## Design Choice

### VGG like model
### Inception like model

## Final Architecture:
<p align="center">
<img src="media/final_Network_arch.png" width="450" height="850">
</p>

## Training
<p align="center">
<img src="media/train_accuracy.png" width="400" height="400">
<img src="media/train_loss.png" width="400" height="400">
</p>

## Results
<p align="center">
<img src="media/vgg_valid.jpg" width="400" height="400">
<img src="media/inception_valid.jpg" width="400" height="400">
</p> 

<p align="center">
<img src="media/result1.gif" width="400" height="400">
<img src="media/result2.gif" width="400" height="400">
</p> 
## Lessons:
* Something
