# https://www.tensorflow.org/tutorials/keras/classification?hl=ko

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("tensorflow version : ", tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("shape of train_images -> ", train_images.shape)
print("len of train_labels -> ", len(train_labels))
print("train_labels -> ", train_labels)

print("shape of test_images -> ", test_images.shape)
print("len of test_labels -> ", len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("\nshape of train_images -> ", train_images[0].shape)
print("train_images -> \n", train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


print("array of predictions[0] -> ", predictions[0])
print("predicted labels -> ", np.argmax(predictions[0]))
print("labels -> ", test_labels[0])



img = test_images[1]
print("\nshape of original image -> ", img.shape)
img = (np.expand_dims(img, 0))
print("shape of expanded image -> ", img.shape)

predictions_single = probability_model.predict(img)
print("array of predictions_single -> ", predictions_single)
print("predicted labels -> ", np.argmax(predictions_single[0]), " (", class_names[np.argmax(predictions_single[0])], ")")