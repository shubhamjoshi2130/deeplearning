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

width=512
height=512
BATCH_SIZE=5
NUM_PARALLEL_CALLS_DS=4


re_siz=Resizing(width=width,height=height)

def resize(x_inp,y):
    #tf.print('{{{{{{{{{{{{{{{{{{{{{{{{',tf.shape(x_inp))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Type:', x_inp)
    return tf.py_function( re_siz, [x_inp] , Tout=(tf.float32)),y

def augment_img(inp):
    rotate=RandomRotation(.1,seed=42)(inp)
    zoom=RandomZoom(.1,seed=42)(rotate)
    flip=RandomFlip(mode="horizontal_and_vertical",seed=42)(zoom)
    return flip

def preprocessing(x_inp,y):
    #tf.print('{{{{{{{{{{{{{{{{{{{{{{{{',tf.shape(x_inp))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Type:', x_inp)
    return tf.py_function( augment_img, [x_inp] , Tout=(tf.float32)),y

train_ds=train_ds.map(resize,num_parallel_calls=NUM_PARALLEL_CALLS_DS).batch(BATCH_SIZE).map(preprocessing,num_parallel_calls=NUM_PARALLEL_CALLS_DS).prefetch(1)
validation_ds=validation_ds.map(resize,num_parallel_calls=NUM_PARALLEL_CALLS_DS).batch(BATCH_SIZE).prefetch(1)

print('##################################', next(iter(train_ds)))

#%%
#fig = tfds.show_examples(train_ds, train_info)
print(train_info)



#%%
# Model building
# model_input=Input(shape=(None,None,3),name='image')
# x=Resizing(height=height,width=width)(model_input)
# x=Rescaling(scale=1/255.)(x)
# x=Conv2D(32,3,padding='same')(x)
# x=Activation(activation='relu')(x)
# x=MaxPool2D(pool_size=(2,2))(x)
# x=Dropout(0.2)(x)
#
# x=Conv2D(32,3,padding='same')(x)
# x=Activation(activation='relu')(x)
# x=MaxPool2D(pool_size=(2,2))(x)
# x=Dropout(0.2)(x)
#
# x=Conv2D(64,3,padding='same')(x)
# x=Activation(activation='relu')(x)
# x=MaxPool2D(pool_size=(2,2))(x)
# x=Dropout(0.2)(x)
#
# x=Conv2D(128,3,padding='same')(x)
# x=Activation(activation='relu')(x)
# x=MaxPool2D(pool_size=(2,2))(x)
# x=Dropout(0.2)(x)
#
# x=Conv2D(128,3,padding='same')(x)
# x=Activation(activation='relu')(x)
# x=MaxPool2D(pool_size=(2,2))(x)
# x=Dropout(0.2)(x)
#
# x=Flatten()(x)
# x=Dense(512,activation='relu')(x)
# x=Dropout(0.2)(x)
# model_out=Dense(196)(x)
# model=Model(inputs=model_input,outputs=model_out)
# model.summary()

model=tf.keras.applications.EfficientNetB4(
        include_top=False, weights='imagenet',
        input_shape=(height,width,3), pooling=None, classes=196,
        classifier_activation='softmax'
    )
x=Flatten()(model.layers[-1].output)
x=Dense(512,activation='relu')(x)
output_model=Dense(196,activation='softmax')(x)
model=Model(inputs=model.inputs,outputs=output_model)

model.summary()


#********************************************************************************
INITIAL_EPOCH=0

save_model_chkpnt=tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(),'/model/model{epoch:02d}.hdf5'),
                                                     monitor='val_accuracy',
                                                     verbose=0,
                                                     save_best_only=True,
                                                     save_freq='epoch')

reduce_on_plateau=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=0,
                                    mode='auto', min_delta=0.01)

logger=tf.keras.callbacks.CSVLogger(
    os.path.join(os.getcwd(),'/log.csv'), separator=',', append=True
)

print(os.getcwd())

if INITIAL_EPOCH!=0:
    del model
    model=tf.keras.models.load_model(os.path.join(os.getcwd(),'/model/model' + f'{INITIAL_EPOCH:02d}' + '.hdf5'))
    logger.set_params(append=False)
    model.summary()
#********************************************************************************

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epoches=200
model.fit(train_ds,
          validation_data=validation_ds,
          epochs=epoches,
          initial_epoch=INITIAL_EPOCH,
          callbacks=[save_model_chkpnt,reduce_on_plateau,logger]
         )