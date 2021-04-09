import flask
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.preprocessing import minmax_scale
import random
import cv2
from werkzeug.utils import secure_filename 
from imgaug import augmenters as iaa
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation

app = flask.Flask(__name__)
app.config["DEBUG"] = True

classes_to_predict = [0, 1, 2, 3, 4]
image_size = 512
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
data_augmentation_layers = tf.keras.Sequential(
[
    layers.experimental.preprocessing.RandomCrop(height=image_size, width=image_size),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.25),
    layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
    layers.experimental.preprocessing.RandomContrast((0.2,0.2))
])

test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2,0.2))
    ]
)

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    efficientnet = EfficientNetB3( 
                              include_top=False, 
                              input_shape=input_shape, 
                              drop_connect_rate=dropout_rate)

    inputs = Input(shape=input_shape)
    augmented = data_augmentation_layers(inputs)
    efficientnet = efficientnet(augmented)
    pooling = layers.GlobalAveragePooling2D()(efficientnet)
    dropout = layers.Dropout(dropout_rate)(pooling)
    outputs = Dense(len(classes_to_predict), activation="softmax")(dropout)
    model = Model(inputs=inputs, outputs=outputs)
        
    model.summary()
    model.load_weights("trained_model.h5")

def scan_over_image(img_path, crop_size=512):
    '''
    Will extract 512x512 images covering the whole original image
    with some overlap between images
    '''
    
    img = Image.open(img_path)
    img_height, img_width = img.size
    img = np.array(img)
    
    y = random.randint(0,img_height-crop_size)
    x = random.randint(0,img_width-crop_size)

    x_img_origins = [0,img_width-crop_size]
    y_img_origins = [0,img_height-crop_size]
    img_list = []
    for x in x_img_origins:
        for y in y_img_origins:
            img_list.append(img[x:x+crop_size , y:y+crop_size,:])
  
    return np.array(img_list)

def predict_and_vote(image_filename, folder, TTA_runs=4):
    '''
    Run the model over 4 local areas of the given image,
    before making a decision depending on the most predicted
    disease.
    '''
    
    #apply TTA to each of the 4 images and sum all predictions for each local image
    localised_predictions = []
    local_image_list = scan_over_image(folder+image_filename)
    for local_image in local_image_list:
        duplicated_local_image = tf.convert_to_tensor(np.array([local_image for i in range(TTA_runs)]))
        augmented_images = test_time_augmentation_layers(duplicated_local_image)
        
        predictions = model.predict(augmented_images)
        localised_predictions.append(np.sum(predictions, axis=0))
    
    #sum all predictions from all 4 images and retrieve the index of the highest value
    global_predictions = np.sum(np.array(localised_predictions),axis=0)
    final_prediction = np.argmax(global_predictions)
    
    return final_prediction

load_model()


@app.route('/predict', methods=['POST'])
def predict():
    
    folder= "content/"
    # print(request.__dict__)
    f = request.files['file']
    
    now = round(time.time() * 1000)
    incoming_file_name = f.filename
    filename = f'content/{now}_{incoming_file_name}'
    f.save(filename)
    
    class_value = predict_and_vote(f'{now}_{incoming_file_name}', folder)
    return str(class_value)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Cassava API"

