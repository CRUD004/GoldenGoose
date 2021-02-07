import numpy as np
import pandas as pd
import tensorflow as tf

import define

def getData(code):
    filename = define.dataPathFormat.format(code)
    data = pd.read_excel(filename)[::-1]
    data = data.reset_index()

    #트레이닝 셋 만들기
    data_size = len(data)
    x_train = data.loc[:data_size-2, ['개인','외국인','기관계']].to_numpy()

    #트레이닝 셋 라벨
    y_element1 = data.loc[:data_size-2,['종가']].to_numpy()
    y_element2 = data.loc[1:,['고가']].to_numpy()
    y_element3 = y_element1*(1+define.target/100)

    y_train = np.greater_equal(y_element3, y_element2)

    return x_train, y_train

def train(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30)

    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    x_train, y_train = None, None
    x_test, y_test = None, None

    for code in define.stockCodes[:-4]:
        print("load data:",code)
        x,y = getData(code)
        if x_train is not None:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
        else:
            x_train = x
            y_train = y

    for code in define.stockCodes[-4:]:
        print("load data:",code)
        x,y = getData(code)
        if x_test is not None:
            x_test = np.concatenate((x_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)
        else:
            x_test = x
            y_test = y

    train(x_train, y_train, x_test, y_test)