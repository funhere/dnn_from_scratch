# Deep Neural Network

## About
  Python implementations of the fundamental vanilla neural network models and algorithms from scratch.

## Installation    
    ```
    $ pip3 install -r requirements.txt
    ```

## Vanilla Neural Network On Iris Dataset
    $ python vnn_demo.py

    +-----+
    | VNN |
    +-----+
    Input Shape: (4,)
    +----------------------+------------+--------------+
    | Layer Type           | Parameters | Output Shape |
    +----------------------+------------+--------------+
    | Dense                | 2560       | (512,)       |
    | Activation (Sigmoid) | 0          | (512,)       |
    | Dense                | 262656     | (512,)       |
    | Activation (Sigmoid) | 0          | (512,)       |
    | Dropout              | 0          | (512,)       |
    | Dense                | 262656     | (512,)       |
    | Activation (Sigmoid) | 0          | (512,)       |
    | Dropout              | 0          | (512,)       |
    | Dense                | 262656     | (512,)       |
    | Activation (Sigmoid) | 0          | (512,)       |
    | Dropout              | 0          | (512,)       |
    | Dense                | 1539       | (3,)         |
    | Activation (Softmax) | 0          | (3,)         |
    +----------------------+------------+--------------+
    Total Parameters: 792067

    Training: 100% [--------------------------------------------------] Time: 0:00:19
    Accuracy: 1.0

## Implement
### Neural Networks consist of the following components:
  + An input layer, x
  + An arbitrary amount of hidden layers
  + An output layer, ŷ
  + A set of weights and biases between each layer, W and b
  + A choice of activation function for each hidden layer, σ. I’ll use a Sigmoid activation function.

### Each iteration of the training process consists of the following steps:
  + Calculating the predicted output ŷ, known as feedforward
  + Updating the weights and biases, known as backpropagation
 
### The main sequential process of train and test:
```python
class NeuralNetwork():
    
    """ Single gradient update over one batch of samples """
    def train_on_batch(self, X, y):
        y_pred = self._forward(X)

        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward(loss_grad=loss_grad)

        return loss, acc
       
    """ Evaluates the model over a single batch of samples """    
    def test_on_batch(self, X, y):
        y_pred = self._forward(X, training=False)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc    
```

### Feedforward
 Feedforward on each layers.
 Example of Dense-layer:

```python
class Dense(Layer):
    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)
        
    def forward(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0
```

### Loss Function
 Use a CrossEntropy as our loss function. 
 to find the best set of weights and biases that minimizes the loss function in training.
```python
class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
```

### Backpropagation
 Update the weights and biases by increasing/reducing with gradient descent on each layers.
 Chain rule for calculating derivative of the loss function with respect to the weights. 
 
 Example of Dense-layer:
```python
class Dense(Layer):
    def backward(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        #import ipdb; ipdb.set_trace()
        
        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad
```


## Source Structure
  + [Model Types]
    * [Vanilla Neural Network](vnn_demo.py)
  + [Neural Network](deep_learning/neural_network.py)
  + [Layers](deep_learning/layers.py)
    * Activation Layer
    * Batch Normalization Layer
    * Dropout Layer
    * Flatten Layer
    * Fully-Connected (Dense) Layer
  + [Optimizers](deep_learning/optimizers.py) 
    * Stochastic Gradient Descent
    * Adagrad
    * Adadelta
    * RMSprop
    * Adam
  + [Activation Functions](deep_learning/activation_functions.py) 
    * Sigmoid
    * Softmax
    * ReLU
    * LeakyReLU 
    * ELU   
    * SELU
  + [Loss Functions](deep_learning/loss_functions.py)
    * Square Loss
    * Cross Entropy Loss

## Data Set
   Iris Data Set: <https://archive.ics.uci.edu/ml/datasets/iris>
   
## Reference
  + [Optimizers]: <http://sebastianruder.com/optimizing-gradient-descent/index.html>
  + [Activations]: <https://en.wikipedia.org/wiki/Activation_function>
  
## ToDo
  1. Fine tunning with GridSearch;
  2. Ensemble
