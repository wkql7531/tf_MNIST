import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def trainData_pretreat(train_images, train_labels, test_images, test_labels):
    Processed_train_images = train_images.astype('float32')/255.0
    Processed_test_images = test_images.astype('float32')/255.0
    Processed_train_labels = tf.keras.utils.to_categorical(train_labels)
    Processed_test_labels = tf.keras.utils.to_categorical(test_labels)

    return Processed_train_images, Processed_train_labels, Processed_test_images, Processed_test_labels


def imgData_pretreat(img,show=True):
    Processed_img = img.resize((28,28))
    Processed_img_data = np.array(Processed_img)
    if show:
        plt.imshow(Processed_img_data, cmap = "binary")
        plt.show()
    Processed_trans_img = Processed_img_data.transpose(2,0,1)
    Processed_trans_test_img = Processed_trans_img[0].reshape((1,28,28))
    Processed_trans_test_img = Processed_trans_test_img.astype('float32')/255.0

    return Processed_trans_test_img



if __name__ == "__main__":
    pass