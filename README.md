# versioncnn1
pneumoniaCNN.ipynb : done with google Colaboratory 
  Make sure to activate GPU (Modifier-> Paramètres du Notebook-> Accélérateur matériel-> GPU -> Enregistrer)
  Detects Pneumonia : val_acc': 0.6343749761581421, 'val_loss': 0.6910778284072876
test.py : done with IDLE 
  contains some functions needed for the dataset ( converting images from dicom to jpg/ splitting..)
  Python version : 3.10
  pydicom                      2.3.0
  Pillow                       9.0.0
  numpy                        1.22.1
  opencv-contrib-python        4.6.0.66
  opencv-python                4.5.5.64
  matplotlib                   3.5.1
  scipy                        1.7.3
# Dataset
Kaggle dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

all images should be resized the same to be able to work with CNN 

images were transformed into size 64 * 64 (they were way larger but there wasn't enough memory )

# Choosing batch size
start with a number and keep doubling it until the training time decreases ( if there is a runtime error , decrease the batch size)
