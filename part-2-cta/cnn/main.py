import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
from tensorflow import keras
import warnings


import tensorflow as tf
from tensorflow import keras
from CTA.dataset import Dataset
from CTA.visualize import plot
from CTA.model import Model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def data_wrangling():
    '''Function that processes the data and leaves everything ready for the training and testing of different models'''

    from CTA.dataset import Dataset
    names=['gamma', 'electron', 'proton', 'helium', 'iron', 'nitrogen', 'silicon']
    lista=[]
    for i in range (7):
        lista.append(load('data/npy/'+names[i]+'_n.npy'))
        #lista[i]=pd.DataFrame(data=lista[i])

    # Initialize the class that works with the dataset
    Dataset = Dataset(lista, names)
    dset, dat = Dataset.load_data()

    # In case it's needed, preprocess the data
    #_,dat, dset = Dataset.preprocess(calculus = False)

    # Save the Images and shift gamma
    Dataset.save_images(dset)
    #Dataset.shift('gamma')

    # Plot and save the data images
    print('plotting data...')
    plot(lista, names)
    return None


def NN(load_model):
    '''Function that creates a Neural Network model. Does the training, testing and generates statistics'''

    from CTA.model import Model
    names = ['gamma_image_moved', 'electron_image', 'proton_image', 'helium_image', 'nitrogen_image', 'iron_image',
             'silicon_image']
    dataset = []
    for i in range(7):
        dataset.append(load('data/npy/' + names[i] + '.npy'))

    Model = Model(dataset, names, numClasses=7)

    x_train, y_train, x_test, y_test = Model.split()

    if load_model:
        print('loading model...')
        model = tf.keras.models.load_model('data/model/NN/all_all.model')
        history = pd.read_csv('data/history/NN/history.csv', delimiter=',')
    else:
        print('train: ')
        history, model = Model.train(x_train, y_train, x_test, y_test, epochs=7)


    print('test: ')
    val_acc, confusion_matrix, classification_report = Model.test(model, x_test, y_test)
    print(confusion_matrix)
    print(classification_report)
    print('generating plots...')
    Model.hist_plots(history, title='NN')

    return None


def CNN(load_model):
    '''Function that creates a Convolutional Neural Network model. Does the training, testing and generates statistics'''

    from CTA.model import Model
    # Load the data for the model
    from numpy import load
    names = ['gamma_image_moved', 'electron_image', 'proton_image', 'helium_image', 'nitrogen_image', 'iron_image',
             'silicon_image']
    dataset = []
    for i in range(7):
        dataset.append(load('data/npy/' + names[i] + '.npy'))

    Model = Model(dataset, names, numClasses=7)
    x_train, y_train, x_test, y_test = Model.split()
    if load_model:
        print('loading model...')
        cnn_model = tf.keras.models.load_model('data/model/CNN/all_all.model')
        history = pd.read_csv('data/history/CNN/history.csv', delimiter=',')
    else:
        shape = x_train.shape
        cnn_model = Model.cnn_create_model()
        cnn_model.build(shape)
        cnn_model.summary()
        history = cnn_model.fit(x_train, y_train, epochs=2, batch_size=128, validation_data=(x_test, y_test))
        history = pd.DataFrame(history.history)
        history.to_csv("data/history/CNN/history.csv", index=False)
        # SAVE THE MODEL
        model.save('data/model/CNN/CNN.model')

    val_acc, confusion_matrix, classification_report = Model.test(cnn_model, x_test, y_test)
    print(confusion_matrix)
    print(classification_report)

    Model.hist_plots(history, title=CNN)

    #In case grid search wants to be applied
    #results = Model.grid_search(x_train, y_train)
    return None

def predict():
    '''Function that generates a plot with predictions for the NN model'''

    from CTA.model import Model
    # Load the data for the model
    names = ['gamma_image_moved', 'electron_image', 'proton_image', 'helium_image', 'nitrogen_image', 'iron_image',
             'silicon_image']
    dataset = []
    for i in range(7):
        dataset.append(load('data/npy/' + names[i] + '.npy'))

    Model = Model(dataset, names, numClasses=7)
    x_train, y_train, x_test, y_test = Model.split()

    # Load the model:
    model = tf.keras.models.load_model('data/model/NN/all_all.model')

    class_names = ['gamma', 'electron', 'proton', 'helium', 'nitrogen',
                   'iron', 'silicon']
    predictions = model.predict(x_test)
    np.argmax(predictions[0])
    y_pred = model.predict_classes(x_test)

    # For instance, here I can select wrong classification for a certain particle
    index=np.where((y_test==6)&(y_pred!=y_test)) [0]
    index2=np.where(y_test==2)[0]
    # index=np.concatenate([index1,index2])
    index=np.sort(index)
    incorrect = np.where(y_pred!=y_test)[0]
    correct = np.where(y_pred==y_test)[0]

    # Do the plot
    y_test= y_test.astype(int)
    num_rows = 4
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      Model.plot_image(i, predictions[index][i], y_test[index], x_test[index], class_names)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      Model.plot_value_array(i, predictions[index][i],  y_test[index])
    plt.tight_layout()
    plt.savefig("images/plots/predictions.png")
    plt.show()
    return None




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", help="Data Wrangling", default=False, action="store_true")
    parser.add_argument("-NN", "--NN", help="Neural Network model", default=False, action="store_true")
    parser.add_argument("-l", "--load_model", help="load an existing model", default=False, action="store_true")
    parser.add_argument("-CNN", "--CNN", help="CNN model", default=False, action="store_true")
    parser.add_argument("-p", "--predict", help="predict images", default=False, action="store_true")

    args = parser.parse_args()
    #warnings.filterwarnings("ignore")
    if args.data:
        data_wrangling()
    if args.NN:
        NN(args.load_model)
    if args.CNN:
        CNN(args.load_model)
    if args.predict:
        predict()




















