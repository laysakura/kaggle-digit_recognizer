import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils


def main():
    #data_dir = '../data_mini'
    data_dir = '../data'
    out_dir = '..'
    input_dim = 784
    nb_hidden_units = [128, 128, 128, 128]
    batch_size = 128
    nb_classes = 10
    nb_epoch = 20  # 1つのテストデータを何回学習するか
    dropout_ratio = 0.2
    validation_ratio = 0.2
    weight_decay = 1e-4

    # 訓練集合、テスト集合の準備
    X_train = np.loadtxt(data_dir + '/train.csv', delimiter=',', skiprows=1, usecols=range(1, input_dim + 1))
    y_train = np.loadtxt(data_dir + '/train.csv', delimiter=',', skiprows=1, usecols=[0])
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    X_test = np.loadtxt(data_dir + '/test.csv', delimiter=',', skiprows=1, usecols=range(input_dim))

    # モデル構築・初期化
    model = Sequential()

    model.add(Dense(nb_hidden_units[0], input_shape=(input_dim,), init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_ratio))

    for l in range(1, len(nb_hidden_units)):
        model.add(Dense(nb_hidden_units[l], init='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_ratio))

    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=weight_decay, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    # 学習・検証
    model.fit(
        X_train, Y_train, nb_epoch=nb_epoch,
        batch_size=batch_size, validation_split=validation_ratio,
        show_accuracy=True)

    # テスト集合に対する予測結果の出力
    classes = model.predict_classes(X_test, batch_size=batch_size)
    print(classes)
    np.savetxt(
        out_dir + '/ans.csv', np.dstack((range(1, len(classes) + 1), classes))[0],
        delimiter=',', header='ImageId,Label', comments='', fmt='%i')


if __name__ == '__main__':
    main()
