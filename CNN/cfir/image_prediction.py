#%%
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,Input,MaxPool2D,Flatten
from tensorflow.keras import Model
from tensorflow.keras import datasets
import numpy as np

#%%
# Load dataset
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
(train_images,test_images)=train_images/255.,test_images/255.

#%%
print('Train shape:',train_images.shape)
print('Test shape:',train_labels.shape)
print(np.unique(train_labels))

#%% prepare the model
input_=Input(shape=(32,32,3))
x=Conv2D(32,(3,3),activation='relu')(input_)
x=MaxPool2D((2,2))(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=MaxPool2D((2,2))(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=Flatten()(x)
x=Dense(64,activation='relu')(x)
out_=Dense(10)(x)

model=Model(inputs=input_,outputs=out_)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

#%% train the model
model.fit(train_images,train_labels,epochs=20,validation_data=(test_images,test_labels))

#%%
# You can use numba library to release all the gpu memory
# pip install numba
from tensorflow.keras import backend as K
K.clear_session()

#%%
import tensorflow as tf
print(tf.__version__)
