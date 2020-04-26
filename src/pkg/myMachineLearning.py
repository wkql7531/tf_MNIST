import tensorflow as tf

import pkg.myPretreat as myPretreat

def mnistMachine_learning(train_images, train_labels, test_images, test_labels):

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
    train_images, train_labels, test_images, test_labels = myPretreat.trainData_pretreat(train_images, train_labels, test_images, test_labels)

    #기계학습
    "#batch_size = 한번학습할때 128장씩 학습 배치싸이즈가 없으면 오래걸림"
    model.fit(train_images, train_labels, epochs=10, batch_size= 128)
    model.evaluate(test_images, test_labels)

    return model

if __name__ == "__main__":
    pass