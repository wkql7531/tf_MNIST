import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL 
import random

import pkg.myPretreat as myPretreat
import pkg.myMachineLearning as myMachineLearning

def pltshow(output):
        #입력받은 숫자 numpy.ndarry형의 ndarry찾기
        finding_index = np.where(train_labels == np.argmax(output))
        #index 저장
        img_index = finding_index[0][int(random.randrange(int(finding_index[0].shape[0])))]
        #해당 index의 img출력
        plt.imshow(train_images[img_index], cmap = "binary")
        #img보기
        plt.title("train_image")
        plt.show()

if __name__ == "__main__":
    #설정 os.getcwd()에 필터로 코딩가능
    Img_path = "C:/Users/장희동/Desktop/Project_Tensorflow/src/imgs/"

    #학습데이터 다운로드
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    #머신러닝
    model = myMachineLearning.mnistMachine_learning(train_images, train_labels, test_images, test_labels)

    #test_imgs
    trans_train_images, trans_train_labels, trans_test_images, trans_test_labels = myPretreat.trainData_pretreat(train_images, train_labels, test_images, test_labels)
    print( trans_test_labels[0], np.argmax(model.predict(trans_test_images[0].reshape(1,28,28))))

    for i in range(10):
        #이미지 열기
        img = PIL.Image.open( Img_path + str(i) +".jpg")

        #img 전처리
        trans_test_img = myPretreat.imgData_pretreat(img,show=True)
        
        #결과출력
        output = model.predict(trans_test_img)
        pltshow(output)
        print("result "+ str(i) + " :" , np.argmax(output))
