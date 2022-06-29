import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class Model:

    def __init__(self, dset, names, numClasses):
        self.dset = dset
        self.names = names
        self.numClasses = numClasses

    def split(self):
        # create the targets
        target = []
        for i in range(7):
            target.append(i * np.ones(self.dset[i].shape[0]))

        target = np.concatenate((target[0], target[1], target[2], target[3], target[4], target[5], target[6]), axis=0)
        # put the dataset together
        data = np.concatenate((self.dset[0], self.dset[1], self.dset[2], self.dset[3], self.dset[4], self.dset[5], self.dset[6]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=3)

        # input image dimensions
        imgRows, imgCols = 55, 93
        numChannels = 1
        samples = X_train.shape[0]

        x_train = X_train
        x_test = X_test

        # #to put  the channels at the beginning
        # x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        # input_shape = (1, img_rows, img_cols)

        # to put  the channels at the end
        x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, numChannels)
        x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, numChannels)
        input_shape = (imgRows, imgCols, numChannels)

        # convert class vectors to binary class matrices
        y_train_c = keras.utils.to_categorical(y_train, self.numClasses)
        y_test_c = keras.utils.to_categorical(y_test, self.numClasses)

        return x_train, y_train, x_test, y_test


    def train(self, x_train, y_train, x_test, y_test, epochs):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.numClasses, activation=tf.nn.softmax))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        history = pd.DataFrame(history.history)
        history.to_csv("data/history/NN/history.csv", index=False)
        # SAVE THE MODEL
        model.save('data/model/NN/NN.model')


        return history, model

    def test(self, model, X_test, y_test):

        val_loss, val_acc = model.evaluate(X_test, y_test)
        y_pred = model.predict_classes(X_test)

        return val_acc, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred)

    def hist_plots(self, history, title):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.savefig("data/history/"+title+"/loss.png")
        plt.show()

        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.savefig("data/history/" + title + "/accuracy.png")
        plt.show()

        return None

    def cnn_create_model(self, padding='same', stride=2, pool=2, filters=64, kernel=5, hidden_layers=2, neurons=20,
                     optimizer='Adadelta', learn_rate=0.001, init_mode='glorot_normal', activation='relu', dropout_rate=0.4,
                     weight_constraint=None, weight_regularizer=None, batchn=0, conv=2, n_filters=2):  # parameters here are default

        # initialize the model
        model = Sequential()

        for i in range(conv):
            num = int(round(filters * (2 ** (i - 1))))
            for i in range(n_filters):
                model.add(Conv2D(num, kernel, padding=padding, activation=activation, kernel_initializer=init_mode,
                                 kernel_constraint=weight_constraint, kernel_regularizer=weight_regularizer))
            model.add(MaxPooling2D(pool_size=(pool, pool), strides=(stride, stride)))
            # model.add(Dropout(dropout_rate))
        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        if (batchn == 1):
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        for i in range(hidden_layers):
            model.add(Dense(neurons, kernel_initializer=init_mode, activation=activation,
                            kernel_regularizer=weight_regularizer))
            if (batchn == 1):
                model.add(BatchNormalization())  # , use_bias=False
            model.add(Dropout(dropout_rate))
            # lastly, define the soft-max classifier
        model.add(Dense(self.numClasses, activation='softmax'))

        if (optimizer == 'SGD'):
            optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        elif (optimizer == 'RMS'):
            optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        elif (optimizer == 'Adagrad'):
            optimizer = keras.optimizers.Adagrad(learning_rate=0.01)
        elif (optimizer == 'Adadelta'):
            optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
        elif (optimizer == 'Adamax'):
            optimizer = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        elif (optimizer == 'Nadam'):
            optimizer = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        elif (optimizer == 'Adam'):
            optimizer = keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def plot_image(self,i, predictions_array, true_label, img, class_names):
        '''Function that plots the predictions and save the plots
        Inputs: predictions_array:array of prediction, true_label: array of the real labels of the predictions
        img: image to plot, class_names: name of the labels'''

        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)  # , cmap=plt.cm.binary

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)
        return None

    def plot_value_array(self,i, predictions_array, true_label):
        '''Function used along with plot_image to plot the values of the arrays. It takes the array of predictions and
        the true labels'''

        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(7), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
        return None

    def grid_search(self, x_train, y_train):
        '''Function used to perform grid_search. It is set to be applied to the cnn model. Below there's a list of
        parameters than can be tuned'''

        # PARAMETERS THAT CAN BE TUNED
        '''batch_size = [64]#[128,264]
        epochs = [500]
        optimizer = ['Adam','SGD','Adadelta']#['Nadam','Adagrad','SGD', 'RMSprop', 'Adagrad', 'Adadelta',  'Adamax']
        learn_rate = [0.001,0.0001]#[0.001, 0.01, 0.1, 0.2, 0.3]
        #In this case momentum is only used for SGD for exmple. For optimizers like Adam you tune rather learning rate
        momentum = [0]#[0.0, 0.2, 0.4, 0.6, 0.8, 0.9]  
        activation = ['relu','elu']#['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        init_mode = ['glorot_normal','lecun_uniform']#['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                        # 'glorot_uniform', 'he_normal', 'he_uniform']
        neurons = [15,128,500, 1000] #[5, 10, 15, 20, 25, 30]
        hidden_layers= [1,2]
        padding=['valid', 'same']
        filters=[32,64]
        weight_regularizer=[regularizers.l2(0.001) , None]
        weight_constraint=[MaxNorm(), None]
        batchn=[0,1]
        n_filters=[1,2]
        kernel = [3, 5]
        stride = [1, 2]
        pool = [2, 4]
        dropout_rate = [0.4, 0.6]
        conv = [1, 2]'''

        kernel = [3, 5]
        stride = [1, 2]
        pool = [2, 4]
        dropout_rate = [0.4, 0.6]
        conv = [1, 2]

        param_grid = dict(conv=conv, dropout_rate=dropout_rate, kernel=kernel, pool=pool, stride=stride)

        # Create the model for grid search
        # If I add any param here has to be also in the model(there has to be a default value)#the values that I set here are definitive
        model = KerasClassifier(build_fn=self.cnn_create_model, epochs=20, batch_size=64, verbose=2)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=2)
        # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, cv=2, n_iter=12, verbose=2)

        grid_result = grid.fit(x_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # SAVE RESULTS
        results = pd.DataFrame(grid_result.cv_results_)
        results.to_csv('data/grid_search/classification.csv', index=False)
        # LOAD RESULTS
        # results = pd.read_csv('data/grid_search/classification.csv', delimiter=',')



        # PLOTS
        # first get the name of the columns
        lista = []
        for col in results.columns:
            lista.append(col)
        count = 0
        for i in results:
            # plot learning curves
            pyplot.title('accuracy')
            pyplot.xlabel(str(lista[count]))
            pyplot.ylabel('Score')
            pyplot.scatter(results[i], results['mean_test_score'])
            pyplot.show()
            count = count + 1

        return results
















