#https://www.tensorflow.org/tutorials/keras/classification?hl=ko
#코드 출처

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure()
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([]) #x축 눈금 값
#     plt.yticks([]) #y축 눈금 값
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(19, activation='softmax')
# ])



# model.compile(optimizer='adam',
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5)

model = load_model('fashion_mnist_mlp_model.h5')
#model.layers[1].get_weights()###이걸로 각각의 layer에 weight에 접근할 수 있다
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nAccuracy of test', test_acc)

predictions = model.predict(test_images)

#model.save('fashion_mnist_mlp_model.h5')

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:   ##이렇게 간단하게..!
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                            100*np.max(predictions_array),
                                            class_names[true_label]),
                                            color=color)
def plot_value_array(i, predictions_array, true_label):
    predictions_arrayi, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(19), predictions_arrayi, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_arrayi)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

