import tensorflow as tf

#x_train 손글씨 숫자이미지 , y_train 이미지가 의미하는 숫자 (6만개)(1만개) 학습용과 정확도 측정용(모델의 입력으로사용된 데이터만 측정잘되는 오버피팅 방지)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#0~255에서 0~1범위로 변경
x_train, x_test = x_train/255.0 , x_test/255.0
#입력 레이어노드 784 히든 레이어노드 128 출력레이어노드 10개인 뉴럴 네트워크를 생성
model = tf.keras.models.Sequential([
    #28*28 배열을 받아 평탄화
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

#옵티마이저와 손실함수 메트릭을 선택
model.compile(
    optimizer='adam',
    #one-hot 엔코딩이 되어있으면 _categorical_crossentropy
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]

)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)