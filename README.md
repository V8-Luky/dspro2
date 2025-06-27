# Silent Speech
In this data science project for the course Data Science Project 2 (DSPRO2) we try to use a neural network to classify sign language gestures. We'll be focusing on the letters of the American Sign Language (ASL). The goal of the project is to evaluate different approaches, likey convolutional neural networks, pretrained vision models and posedetection models, using MediaPipe.

The training of the models can be found as Jupyter Notebooks in the root folder of this repository.

## Team
- Luca Kyburz
- Luca Niederer
- Sevan Sherbetjian

## Dataset
The data used to train our models was taken from the [American Sign Language](https://www.kaggle.com/datasets/kapillondhe/american-sign-language) dataset on Kaggle. However to make our models useful in real life scenarios, we had to apply extensive data augmentation. This dataset was only used as training and validation data.

### Test Dataset
For our test data we used a manually created dataset of 691 images of the same classes (hand gestures) as the [American Sign Language](https://www.kaggle.com/datasets/kapillondhe/american-sign-language) dataset.

## Client Application
Along with our models we also wanted to provide a simple client application that can take our models and do real time classification on ASL signs.