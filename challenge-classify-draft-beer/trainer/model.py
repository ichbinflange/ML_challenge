#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf



def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 4

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 30

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.
    

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """
    

    # TODO: Code of your solution
   

    def dataaugmenter():
        dataugumentation=tf.keras.models.Sequential()
        dataugumentation.add(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'))
        dataugumentation.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2))
        dataugumentation.add(tf.keras.layers.experimental.preprocessing.RandomContrast(0.2))

        return dataugumentation
       
    
    initializer = tf.keras.initializers.HeNormal(seed=123)
    # TODO: Code of your solution

    base_model = tf.keras.applications.MobileNetV2(include_top = False)
    base_model.trainable = False

    #x = dataaugmenter()(input_layer)
    x = base_model(input_layer,training = False)

    x= tf.keras.layers.GlobalAveragePooling2D()(x)
    x= tf.keras.layers.Dropout(0.2)(x)
    x= tf.keras.layers.Dense(units =100,activation='relu')(x)
    outputs = tf.keras.layers.Dense(units =5,activation='softmax')(x)

    model = tf.keras.Model(input_layer,outputs)


    
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),loss ='sparse_categorical_crossentropy',metrics= ['accuracy'])

    # TODO: Return the compiled model
    return model


