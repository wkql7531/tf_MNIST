import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL 

import pkg.myPretreat as my


if __name__ == "__main__":
    #설정
    Img_path = "C:/Users/장희동/Desktop/Project_Tensorflow/imgs/"

    #학습데이터 다운로드
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    #레이어
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    #컴파일
    model.compile(
        optimizer = 'adam',#rmsprop,adam(가장효율이 좋음)
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    #학습용img 전처리
    train_images, train_labels, test_images, test_labels = my.trainData_pretreat(train_images, train_labels, test_images, test_labels)
    
    #기계학습
    "#batch_size = 한번학습할때 128장씩 학습 배치싸이즈가 없으면 오래걸림"
    model.fit(train_images, train_labels, epochs=10, batch_size= 128)
    model.evaluate(test_images, test_labels)

    for i in range(10):
        img = PIL.Image.open( Img_path + str(i) +".jpg")

        #img 전처리
        trans_test_img = my.imgData_pretreat(img,show=False)

        #결과출력
        output = model.predict(trans_test_img)
        print("result "+str(i)+ " :" , np.argmax(output))
