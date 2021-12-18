import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy

from tensorflow.keras import datasets, layers, models

epochs = 10
batch_size = 32
img_size = 256

train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "Train")
test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "Test")

train_data = tf.keras.utils.image_dataset_from_directory(directory=train_path,
                                                         seed=123,
                                                         batch_size=batch_size,
                                                         image_size=(img_size, img_size),
                                                         validation_split=0.2,
                                                         subset="validation")


test_data = tf.keras.utils.image_dataset_from_directory(directory=test_path,
                                                         seed=123,
                                                         image_size=(img_size, img_size))                                                     


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(4))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, epochs=epochs, verbose=1)

test_loss, test_acc = model.evaluate(test_data, verbose=1)