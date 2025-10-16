import os
import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)

main_path = './f1cars/'
img_size = (64, 64)
batch_size = 32

def clean_and_convert_images(folder):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if file.startswith('.'):
                os.remove(path)
                print(f"Obrisan skrivni/sistemski fajl: {path}")
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(path, format='JPEG')
            except Exception as e:
                print(f"Brisanje oštećenog/nepodržanog fajla: {path} ({e})")
                os.remove(path)

clean_and_convert_images(main_path)

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
print(classes)

N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

from keras import layers
from keras import Sequential

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_size[0],
                                                 img_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
  ]
)
N = 10

plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

num_classes = len(classes)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
history = model.fit(Xtrain,
                    epochs=20,
                    validation_data=Xval,
                    verbose=0)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()
labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))
from sklearn.metrics import accuracy_score
print('Tačnost modela je: ' + str(100*accuracy_score(labels, pred)) + '%')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.show()


