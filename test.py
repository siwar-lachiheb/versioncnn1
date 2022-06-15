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

    os.makedirs(root_dir +'/train')
    os.makedirs(root_dir +'/val')
    os.makedirs(root_dir +'/test')

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
#resizing images 
import PIL
import os
from PIL import Image
directories=['d0','d1','d2']
for d in directories:
  for file in os.listdir(d):
    d_img = d+"/"+file
    img = Image.open(d_img)
    img = img.resize((64,64))
    img.save(d_img)
