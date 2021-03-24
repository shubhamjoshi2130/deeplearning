#%%
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,Input,BatchNormalization,Concatenate,Activation,Add
from tensorflow.keras import Model
import tensorflow as tf
import os
import zipfile
import tarfile
import PIL as pil
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import Rescaling,RandomFlip,RandomRotation,RandomZoom
from tensorflow.keras.preprocessing import image_dataset_from_directory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%
import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file(os.path.join(os.getcwd(),'flower_photos'), origin=dataset_url, untar=True)

#%%
# zip_ref=tarfile.open(os.path.join(os.getcwd(),'flower_photos.tgz'),'r')
# zip_ref.extractall(os.path.join(os.getcwd(),'flower_photos'))
# zip_ref.close()


#%%
# Count number of files
data_dir=pathlib.Path(os.path.join(os.getcwd(),'flower_photos/flower_photos'))
lst_img=list(data_dir.glob('*/*.jpg'))
count_files=len(lst_img)
print(count_files)

#%%
# Vizualize some images
plt.imshow(pil.Image.open(lst_img[0]))

#%%
batch_size = 32
img_height = 180
img_width = 180


train_ds=image_dataset_from_directory(
    'flower_photos/flower_photos',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)

num_classes=len(train_ds.class_names)
print(next(iter(train_ds)))

validation_ds=image_dataset_from_directory(
    'flower_photos/flower_photos',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)



#%%
inp=Input(shape=(img_height,img_width,3))
normalization_layer=Rescaling(scale=1/255.)(inp)
rotate=RandomRotation(.1,seed=42)(normalization_layer)
zoom=RandomZoom(.1,seed=42)(rotate)
filp=RandomFlip(mode="horizontal_and_vertical",seed=42)(zoom)

x2=Conv2D(32, 7, padding='same')(filp)
x2=BatchNormalization()(x2)
x2=Activation('relu')(x2)

x3=Conv2D(32, 7, padding='same')(x2)
x3=BatchNormalization()(x3)
x3=Activation('relu')(x3)

x4=Conv2D(32, 7, padding='same')(x3)
x4=BatchNormalization()(x4)
x4=Add()([x3,x4])
x4=Activation('relu')(x4)

x=Conv2D(64, 3, padding='same', activation='relu')(x4)
x=MaxPool2D()(x)
x=BatchNormalization()(x)
x=MaxPool2D()(x)
x=Dropout(0.2)(x)

x=Flatten()(x)
x=Dense(128,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.2)(x)
out=Dense(num_classes)(x)
model=Model(inputs=inp,outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#%%
epochs=60
model.fit(train_ds,
          validation_data=validation_ds,
          epochs=epochs)

    #%%
print(os.getcwd())
