# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os

# Paths
train_path = r"C:\\Users\banuz\\OneDrive\\Desktop\\Traffic\\Traffic\\Train"
test_path = r"C:\\Users\\banuz\\OneDrive\\Desktop\\Traffic\\Traffic\\Test"
test_csv_path = r"C:\\Users\\banuz\\OneDrive\\Desktop\\Traffic\\Traffic\\Test.csv"

# Parameters
data = []
labels = []
height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width * channels

# Load training data
for i in range(classes):
    path = os.path.join(train_path, str(i))
    print(f"Loading from: {path}")
    Class = os.listdir(path)
    for a in Class:
        try:
            image = cv2.imread(os.path.join(path, a))
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(f"Error loading image: {a}")

# Convert to numpy arrays
Cells = np.array(data)
labels = np.array(labels)

# Shuffle data
s = np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells = Cells[s]
labels = labels[s]

# Split into training and validation sets
(X_train, X_val) = Cells[int(0.2 * len(labels)):], Cells[:int(0.2 * len(labels))]
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
(y_train, y_val) = labels[int(0.2 * len(labels)):], labels[:int(0.2 * len(labels))]

# One hot encoding
from keras.utils import to_categorical
y_train = to_categorical(y_train, classes)
y_val = to_categorical(y_val, classes)

# CNN Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))

# Plot Accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# Plot Loss
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Load test data
y_test_df = pd.read_csv(test_csv_path)
labels_test = y_test_df['Path'].values
y_test = y_test_df['ClassId'].values

data_test = []
for f in labels_test:
    image = cv2.imread(os.path.join(test_path, f.replace('Test/', '')))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data_test.append(np.array(size_image))

X_test = np.array(data_test)
X_test = X_test.astype('float32') / 255

# Predictions (updated for TF 2.x)
pred_prob = model.predict(X_test)
pred = np.argmax(pred_prob, axis=1)

# Accuracy
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, pred))

model.save("traffic_classifier.h5")