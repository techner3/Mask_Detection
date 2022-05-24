# Mask_Detection
![my badge](https://img.shields.io/badge/Python-3-blue)
![my badge](https://img.shields.io/badge/Deep-Learning-brightgreen)
![my badge](https://img.shields.io/badge/Flask-App-green)
![my badge](https://img.shields.io/badge/Object-Detection-yellowgreen)
![my badge](https://img.shields.io/badge/TF-OD-orange)
![my badge](https://img.shields.io/badge/-Docker-purple)
![my badge](https://img.shields.io/badge/-GIT-green)

# About The Project

This project has been developed to detect whether a person is wearing mask or wearing it incorrectly or not wearing at all.

# Project Description 

A web app has been developed for this project which takes a image as an input and returns the predictions as a result. The app is dockerized and pushed to dockerhub. Command to pull the image from dockerhub is given below. The object detection model is trained using TFOD 2 (Tensorflow Object detection). The input image is first encode to base64 format after which is it send to the backend where the image is decoded and the prediction is done. And final ly the input image 

# Dataset Used

This dataset is taken from Kaggle and more information about the dataset can be found below.

Dataset : [Link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

# Project Structure


<img width="229" alt="image" src="https://user-images.githubusercontent.com/58848985/169958144-712f2c5f-a765-493a-a075-092098f40b6a.png">


*


# Preview of the Web App

Web App Main Page :

<img width="960" alt="image" src="https://user-images.githubusercontent.com/58848985/161687024-ed21efcd-a887-45c7-8d72-5b3f3cdc1bd6.png">

After Prediction: 

<img width="960" alt="image" src="https://user-images.githubusercontent.com/58848985/161687308-577f42c0-7ee8-4ff9-b876-d34a7907a518.png">


This flask app has been dockerized and pushed it into dockerhub. 
Docker command to pull the image:

```docker pull techner3/maskod```

# Points to Note : 

* The app may take while to load ,Please bear with it 
* Many of the libraries are commented in requirements.txt and libraries needed only for prediction has been installed to reduce slug size while deploying to heroku.
