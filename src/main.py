import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL 

import pkg.myPretreat as myPretreat
import pkg.myMachineLearning as myMachineLearning


if __name__ == "__main__":
    #설정
    Img_path = "C:/Users/장희동/Desktop/Project_Tensorflow/imgs/"

    #학습데이터 다운로드
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    #머신러닝
    model = myMachineLearning.mnistMachine_learning(train_images, train_labels, test_images, test_labels)

    #test_imgs
    trans_train_images, trans_train_labels, trans_test_images, trans_test_labels = myPretreat.trainData_pretreat(train_images, train_labels, test_images, test_labels)
    print( trans_test_labels[0], np.argmax(model.predict(trans_test_images[0].reshape(1,28,28))))

    for i in range(10):
        img = PIL.Image.open( Img_path + str(i) +".jpg")

        #img 전처리
        trans_test_img = myPretreat.imgData_pretreat(img,show=True)

        #결과출력
        output = model.predict(trans_test_img)
        print("result "+ str(i) + " :" , np.argmax(output))
