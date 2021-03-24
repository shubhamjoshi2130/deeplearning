#%%
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,BatchNormalization,Dropout,Input,Activation,Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling,RandomRotation,RandomZoom,RandomFlip,Resizing
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import Cars196
from matplotlib import pyplot as plt
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%
print(os.getcwd())


#%%
train_ds, train_info=tfds.load('cars196', split='train', with_info=True,shuffle_files=True, as_supervised=True)
validation_ds, test_info=tfds.load('cars196', split='test', with_info=True, as_supervised=True)

label_lst=set()
for tr in train_ds:
    label_lst.add(tr[1].numpy())

print(label_lst)