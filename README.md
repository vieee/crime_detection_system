# crime_detection_system

* _The system will use different Computer Vision techniques for video analysis._ 
* _It will monitor CCTV footage for any criminal offenders, violent objects, and suspicious behavior which could lead to crime._
* _[SSD Mobilenet Model](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs), an architecture for concealed object detection, is trained for labeling weapons in the frame._
* _The images captured are processed using Face Detection algorithms to identify human faces._ 
* _Facial Recognition API using libraries in python is implemented to recognize the offenders from criminal records._ 
* _A [ResNet-GRU Model](https://www.researchgate.net/publication/344002214_A_Novel_Fault_Identification_Method_for_Photovoltaic_Array_Via_Convolutional_Neural_Network_and_Residual_Gated_Recurrent_Unit) was trained for human behavior analysis which detects suspicious actions._ 
* _An alert is generated when there are signs of crime and concerned authorities are notified._

[![Tensorflow](https://img.shields.io/badge/tensorflow.js-utilizing-green)](https://js.tensorflow.org/api/latest/)
[![FACE-API](https://img.shields.io/badge/face--api.js-recognition-green)](https://justadudewhohacks.github.io/face-api.js/docs/index.html)
[![Build Status](https://github.com/ageitgey/face_recognition/workflows/CI/badge.svg?branch=master&event=push)](https://github.com/ageitgey/face_recognition/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)


## Steps

###### Requires Conda to run the program
###### conda create --name <envname> --file requirements.txt
###### Activate the environment
###### Change the paths in all the files to find the appropriate data
###### python create_data.py will create the .npy files required for the model to train on
###### Run the SuspiciousModel.ipynb to create the Suspicious Behaviour Detector Models
###### python face.py to collect photos of faces to create the criminal database
###### python encoding.py to create the encoding of the collected photos
###### python multi.py to run all three models in parallel
  
  
## Implementation
  
![Video of the Implementation](https://raw.githubusercontent.com/vieee/crime_detection_system/main/a.webp)
  

## Made By 

* [Harshita Lakhotiya](https://github.com/Harshitalakhotiya)
* [Deepak Yadav](https://github.com/vieee)
* [Rajat Shenoy](https://github.com/rajatshenoy56)

  
  
