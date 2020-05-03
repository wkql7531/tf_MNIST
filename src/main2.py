# ???????????

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

#학습데이터 가져오기
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

while True:
    #숫자 입력받기
    input_num = int(input("0~9까지 숫자를 입력하세요 : "))
    #입력받은 숫자 numpy.ndarry형의 ndarry찾기
    finding_index = np.where(train_labels == input_num)
    print(finding_index)
    #index 저장
    img_index = finding_index[0][int(random.randrange(int(finding_index[0].shape[0])))]
    print(int(finding_index[0].shape[0]))
    #해당 index의 img출력
    plt.imshow(train_images[img_index], cmap = "binary")
    #img보기
    plt.show()