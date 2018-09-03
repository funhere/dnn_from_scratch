
from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import helper functions
from deep_learning import NeuralNetwork
from utils import train_test_split, to_categorical, normalize, Plot
from deep_learning.optimizers import StochasticGradientDescent, Adam, RMSprop, Adagrad, Adadelta
from deep_learning.loss_functions import CrossEntropy
from deep_learning.layers import Dense, Dropout, Activation, BatchNormalization
#import ipdb; ipdb.set_trace()

def main():

    optimizer = Adam()

    #----------------------------
    # VNN(Vanilla Neural Network)
    #----------------------------
    
    # 1). Load the dataset.
    data = datasets.load_iris()
    X = data.data
    y = data.target
    
    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    n_samples, n_features = X.shape
    n_hidden = 512

    # 2). Split dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=1)
    
    # 3). Create the network
    clf = NeuralNetwork(optimizer=optimizer,
                        loss=CrossEntropy,
                        validation_data=(X_test, y_test))

    #import ipdb; ipdb.set_trace()
    
    # 4). Add layers.
    clf.add(Dense(n_hidden, input_shape=(n_features,)))
    clf.add(Activation('sigmoid'))  #[selu;leaky_relu:elu;sigmoid]
    clf.add(Dense(n_hidden))
    clf.add(Activation('sigmoid'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('sigmoid'))
    clf.add(Dropout(0.25))
    clf.add(Dense(n_hidden))
    clf.add(Activation('sigmoid'))
    clf.add(Dropout(0.25))
    clf.add(Dense(3))
    clf.add(Activation('softmax'))

    print ()
    clf.summary(name="VNN")
    
    # 5). Kicks off the training.
    train_err, val_err = clf.fit(X_train, y_train, n_epochs=300, batch_size=256)
    
    # 6). Training and validation error plot
    n = len(train_err)
    training, = plt.plot(range(n), train_err, label="Training Error")
    validation, = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    _, accuracy = clf.test_on_batch(X_test, y_test)
    print ("Accuracy:", accuracy)

    # Reduce dimension to 2D using PCA and plot the results
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    Plot().plot_in_2d(X_test, y_pred, title="VNN", accuracy=accuracy, legend_labels=range(3))


if __name__ == "__main__":
    main()