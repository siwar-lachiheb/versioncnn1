import pydicom 
import matplotlib.pyplot as plt 
import scipy.misc 
import pandas as pd
import numpy as np
import os 
from PIL import Image
import shutil
#CONVERTIR LES IMAGES DICOM EN JPG/PNG ------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)    
    return names

def convert_dcm_jpg(name, path):
    
    im = pydicom.dcmread(path+name)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)

    return final_image


#APPLICATION : UNCOMMENT IF NECESSARY


##path='Data/650458-/' #on spécifie le nom des images qu'on va convertir
##directory='Datajpg/'
##names = get_names(path[:(len(path)-1)])
##for name in names:
##    image = convert_dcm_jpg(name,path)
##    nom=name+'.png'
##    image.save(name+'.png')
##    shutil.move(nom,directory)#transférer les images converties dans un nouveau dossier



#ELIMINER LE  BRUIT DES IMAGES (methode de Non Local Means)---------------------------------------------------------------------------------------------------------------------------------------
import cv2 as cv
def Denoising (img):
    return cv.fastNlMeansDenoising(img,None,3,7,21)
cv.waitKey(0)

#SPLIT DATA INTO TRAINING AND TESTING AND VALIDATING---------------------------------------------------------------------------------------------------------------------------------------------
#https://stackoverflow.com/questions/54263218/cnn-divide-images-into-training-validation-testing?rq=1 #splitting data ffor cnn (needed later)
#import splitfolders
#splitfolders.ratio('Datajpg', output="output", seed=1337, ratio=(0.8, 0.1,0.1))

import os
import numpy as np
import shutil

def splitting (root_dir,src) :
    # # Creating Train / Val / Test folders (One time use)
    #posCls = '/DPN+'
    #negCls = '/DPN-'

    os.makedirs(root_dir +'/train')
    #os.makedirs(root_dir +'/train' + negCls)
    os.makedirs(root_dir +'/val')
    #os.makedirs(root_dir +'/val' + negCls)
    os.makedirs(root_dir +'/test')
    #os.makedirs(root_dir +'/test' + negCls)

    # Creating partitions of the data after shuffeling

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])

    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir+"/train/")

    for name in val_FileNames:
        shutil.copy(name, root_dir+"/val/")

    for name in test_FileNames:
       shutil.copy(name, root_dir+"/test/")
#splitting('output','Datajpg')


#Building the CNN model ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#importing models
import numpy as np
#matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)
#form the cnn model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (180,180,3)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(400,activation ="relu"),
    tf.keras.layers.Dropout(0.3,seed = 2019),
    tf.keras.layers.Dense(300,activation="relu"),
    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(200,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(5,activation = "softmax")   #Adding the Output Layer
])

#print the model summary

model.summary()

# specify optimizers

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
adam=Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])#Optimizer is used to reduce the cost calculated by cross-entropy


#set the data directory and generate images
bs=30         #Setting batch size
train_dir = "C:/Users/lenovo/Desktop/stage/output/train/"   #Setting training directory
validation_dir = "C:/Users/lenovo/Desktop/stage/output/test/"   #Setting testing directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )
# Flow training images in batches of 20 using train_datagen generator
#Flow_from_directory function lets the classifier directly identify the labels from the name of the directories the image lies in
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=bs,class_mode='categorical',target_size=(180,180))
# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=bs,
                                                         class_mode  = 'categorical',
                                                         target_size=(180,180))

###fitting the model
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=150 // bs,
                    epochs=30,
                    validation_steps=50 // bs,
                    verbose=2)
