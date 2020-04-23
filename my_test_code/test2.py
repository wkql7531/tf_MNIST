import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import PIL 

#이미지 다운로드
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#데이터형태 보기
print(train_images.shape)
print(train_labels)

#학습된 데이터 모양보기
digit = train_images[1]

plt.imshow(digit, cmap = "binary")
#plt.show()

#테스트데이터
origin_my_image = test_images[0]

#레이어
model = tf.keras.models.Sequential()
#==model.add(tf.keras.layers.Dense(512, activation="relu", input_shape=(28*28, )))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512, activation="relu", ))
#
model.add(tf.keras.layers.Dense(256, activation="relu", ))
model.add(tf.keras.layers.Dense(128, activation="relu", ))
model.add(tf.keras.layers.Dense(64, activation="relu", ))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(
    optimizer = 'rmsprop',#adam(가장효율이 좋음)
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#데이터 전처리 작업 
#train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
print(train_images.shape)

#test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
print(train_labels)

#batch_size = 한번학습할때 128장씩 학습 배치싸이즈가 없으면 오래걸림
model.fit(train_images, train_labels, epochs=5, batch_size= 128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss)
print(test_acc)

myDigit = origin_my_image

plt.imshow(myDigit, cmap = "binary")
#plt.show()
# (28,28) > (1,28,28)
my_image = myDigit.reshape((1, 28, 28))
output = model.predict(my_image)
print('result:', np.argmax(output))

(t_i, t_l), (test_i, test_l) = tf.keras.datasets.mnist.load_data()
plt.imshow(test_i[1], cmap = "binary")
#plt.show()

output = model.predict(test_i[1].reshape(1, 28, 28))
print('result :' , np.argmax(output))

img = PIL.Image.open("C:/Users/장희동/Desktop/Project_Tensorflow/imgs/9.jpg")
img = img.resize((28,28))

img_data = np.array(img)
#(28,28,3) 3은 rgb값 이여야하는데 (28,28)이다?
print(img_data.shape)
plt.imshow(img_data, cmap = "binary")
plt.show()
#(3,28,28)로 변형
trans_img = img_data.transpose(2,0,1)
print(trans_img.shape)

#np.savetxt("testimg.csv",trans_img[0], delimiter='.')

trans_test_img = trans_img[0].reshape((1,28,28))
trans_test_img = trans_test_img.astype('float32')/255

output = model.predict(trans_test_img)
print("result :" , np.argmax(output))