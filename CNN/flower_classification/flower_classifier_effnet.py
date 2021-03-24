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
from tensorflow.keras import backend as K
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
#print(data_dir)
lst_img=list(data_dir.glob('*/*.jpg'))
count_files=len(lst_img)
#print(count_files)
#print(lst_img)

#%%
# Vizualize some images
# plt.imshow(pil.Image.open(lst_img[0]))

#%%
batch_size = 32
img_height = 180
img_width = 180

def normalization_augmenting(inp):
    normalization_layer=Rescaling(scale=1/255.)(inp)
    rotate=RandomRotation(.1,seed=42)(normalization_layer)
    zoom=RandomZoom(.1,seed=42)(rotate)
    flip=RandomFlip(mode="horizontal_and_vertical",seed=42)(zoom)
    return flip



def preprocessing(x_inp):
    #tf.print('{{{{{{{{{{{{{{{{{{{{{{{{',tf.shape(x_inp))
    return tf.py_function( normalization_augmenting, [x_inp] , Tout=(tf.float32))

with tf.device('/cpu:0'):
    train_ds=image_dataset_from_directory(
        'flower_photos/flower_photos',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    num_classes = len(train_ds.class_names)

    train_ds = train_ds.map(lambda x, y: (preprocessing(x), y)).prefetch(1)

    # print(next(iter(train_ds)))

    validation_ds=image_dataset_from_directory(
        'flower_photos/flower_photos',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height,img_width),
        batch_size=batch_size
    )
    normalization_layer = Rescaling(scale=1 / 255.)
    validation_ds = validation_ds.map(lambda x, y: (preprocessing(x), y)).prefetch(1)


#inp=Input(shape=(img_height,img_width,3))


strategy=tf.distribute.MirroredStrategy()

with strategy.scope():
    model=tf.keras.applications.EfficientNetB0(
        include_top=True, weights=None,
        input_shape=(img_height,img_width,3), pooling=None, classes=num_classes,
        classifier_activation='softmax'
    )



    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    #%%

    INITIAL_EPOCH=0

    save_model_chkpnt=tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(),'flower_classification/model/model{epoch:02d}.hdf5'),
                                                         monitor='val_accuracy',
                                                         verbose=0,
                                                         save_best_only=True,
                                                         save_freq='epoch')

    reduce_on_plateau=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0,
                                        mode='auto', min_delta=0.01)

    logger=tf.keras.callbacks.CSVLogger(
        os.path.join(os.getcwd(),'flower_classification/log.csv'), separator=',', append=True
    )

    np.random.seed(42)

    if INITIAL_EPOCH!=0:
        del model
        model=tf.keras.models.load_model(os.path.join(os.getcwd(),'flower_classification/model/model' + f'{INITIAL_EPOCH:02d}' + '.hdf5'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        logger.set_params(append=False)
        model.summary()

    #print('?????????????????????????????????',f'{0:02d}'.format(str(INITIAL_EPOCH)))

    epochs=600
    history=model.fit(train_ds,
              validation_data=validation_ds,
              epochs=epochs,callbacks=[save_model_chkpnt,reduce_on_plateau,logger],initial_epoch=INITIAL_EPOCH)

    #%%
    print(os.getcwd())
