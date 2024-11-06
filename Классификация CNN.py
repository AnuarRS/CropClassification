import os
import numpy as np
import tensorflow as tf

import pandas as pd


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow import keras
from google.colab import files
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive/MyDrive/trainingAll/training')

train_dataset=image_dataset_from_directory('/content/drive/MyDrive/trainingAll/Обрезки снимков для обучения модели/training')
class_names=train_dataset.class_names
class_names

validation_dataset=image_dataset_from_directory('/content/drive/MyDrive/TrainC/Обрезки снимков для обучения модели/validation')
class_names=validation_dataset.class_names
class_names

plt.figure(figsize=(8, 8))
for images, labels in train_dataset.take(1):
  for i in range (9):
      ax=plt.subplot(3,3,i+1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")

test_dataset=image_dataset_from_directory('/content/drive/MyDrive/TrainC/Обрезки снимков для проверки модели/test')
print(test_dataset)

AUTOTUNE=tf.data.experimental.AUTOTUNE

train_dataset=train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset=validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset=test_dataset.prefetch(buffer_size=AUTOTUNE)

model = Sequential()
model.add(Conv2D(16, (5, 5), padding='same',
                 input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10,
                    verbose=2)

predictions = model.predict(train_dataset)
print(predictions)
np.argmax(predictions[972])
type(class_names[np.argmax(predictions[972])])

final = pd.DataFrame()
for i in range(0, 1122, 1):
  class_type = class_names[np.argmax(predictions[i])]
  df = pd.DataFrame(columns = ['image_index','type' ])
  df.loc[0, 'image_index'] = i
  df.loc[0, 'type'] = class_type
  final = final.append(df)
filepath = 'Егістік дДақылдар.xlsx'

final.to_excel(filepath, index=False)
scores = model.evaluate(test_dataset, verbose=1)

## save to xlsx file

filepath = 'Егістік Дақылдар.xlsx'
df.to_excel(filepath, index=False)

print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))

plt.plot(history.history['accuracy'],
         label='Үйрету жүйесіндегі дұрыс анықтау күші')
plt.plot(history.history['val_accuracy'],
         label='Тексеру жүйесіндегі дұрыс анықтау күші')
plt.xlabel('Үйрену эпохасы')
plt.ylabel('Дұрыс анықтама бөлігі')
plt.legend()
plt.show()

plt.plot(history.history['loss'],
         label='Үйрену жүйесіндегі қателік')
plt.plot(history.history['val_loss'],
         label='Тексеру жүйесіндегі қателік')
plt.xlabel('Үйрену эпохасы')
plt.ylabel('Мүмкін болған қателік')
plt.legend()
plt.show()

model.save("test_360_model.txt")
model.save("test_360_model.h5")
files.download("test_360_model.txt")

