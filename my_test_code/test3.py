import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL 

#학습데이터 다운로드
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


#레이어
model = tf.keras.models.Sequential([
    #28*28 배열을 받아 평탄화
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

#컴파일
model.compile(
    optimizer = 'rmsprop',#adam(가장효율이 좋음)
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#전처리
"train_images = train_images.reshape((60000, 28*28))"
train_images = train_images.astype('float32')/255.0
print(train_images.shape)

"test_images = test_images.reshape((10000, 28*28))"
test_images = test_images.astype('float32')/255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
print(train_labels)

#기계학습
"#batch_size = 한번학습할때 128장씩 학습 배치싸이즈가 없으면 오래걸림"
model.fit(train_images, train_labels, epochs=5, batch_size= 128)
test_loss, test_acc = model.evaluate(test_images, test_labels)

#img 열기
img = PIL.Image.open("C:/Users/장희동/Desktop/Project_Tensorflow/imgs/3.jpg")

#img 전처리
img = img.resize((28,28))
img_data = np.array(img)
"#(28,28,3) 3은 rgb값"
plt.imshow(img_data, cmap = "binary")
plt.show()
"#(3,28,28)로 변형"
trans_img = img_data.transpose(2,0,1)
"print(trans_img.shape)"
"csv파일로 저장"
#np.savetxt("testimg.csv",trans_img[0], delimiter='.')
trans_test_img = trans_img[0].reshape((1,28,28))
trans_test_img = trans_test_img.astype('float32')/255

#결과출력
output = model.predict(trans_test_img)
print("result :" , np.argmax(output))