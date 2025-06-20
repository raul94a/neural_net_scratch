import activations
import numpy as np


class BasicNeuralNetwork():
    layers = []
    def __init__(self, X, Y, sigmoid = activations.sigmoid):
        """
        Constructor for the BasicNeuralNetwork class
        X => Vector of training inputs
        Y => Vector of training ouputs
        """
        self.X = X
        self.Y = Y
        return
    def add_layers(self,layers):
        """
            Add layers to the neural network
            a layer is a tupple of (n,m) where:
                n: rows of the vector
                m: columns of the vector
        """
        for layer in layers:
            self.layers.append(layer)
    def initialize_params(self,layer_dim):
        """
        layer_dim are the dimensions of the neural network layer
        with the form (n,m) where:
            n: rows of the vector
            m: columns of the vector
            
        returns a tupple with the initialized weights and bias

        """
        f,c = layer_dim
        return (np.random.rand(f,c) * 0.1, np.random.rand(f,1) * 0.0000099)
    
    def forward_propagation(self, verbose = False):
        """
            Computes the forward propagation from the input layer to the output layer
        """
        # need to add a cache for the weights and bias!
        X_Transpose = np.transpose(self.X)
        W1,b1 = self.initialize_params(self.layers.pop(0)) 
        a = activations.sigmoid(np.dot(W1,X_Transpose)+ b1)
        if verbose: 
            print(f"Activation of the input layer\n {a}")
        i = 1
        for layer in self.layers:
            w,b = self.initialize_params(layer)
            a = activations.sigmoid(np.dot(w,a) + b)
            
            i += 1

        y_hat = [1 if e > 0.5 else 0 for e in a[0]]
        if verbose:
            print(f"Layer dim: {layer}")
            print(f"Weights: {w}")
            print(f"Activation of the hidden layer {i - 1} \n {a}")
            print(f"Output y_hat: \n{y_hat}")
            
            
    


X = np.array([[-110,-111,-12.5],[-2,0,-1.25], [-105,-0.22, -10.71],[-105,0.22, -10.71],[-0.105,-0.22, -0.71]])
XT = np.transpose(X)
Y = np.array([[0, 1]]) # Note the extra set of brackets for 2D array
# W1 = np.random.randn(5, 3) * 0.01
# b1 = np.random.randn(5, 1) * 0.0001
# WXT = np.dot(W1,XT)
# z1 = WXT + b1
# a1 = activations.sigmoid(z1)

# print('')
# print(f"\nFirst layer Activations\n{a1}")
# print(F"Shape of a1: {a1.shape}")

# # second layer
# W2 = np.random.randn(3, 5) * 0.01
# b2 = np.random.randn(3, 1) * 0.01


# z2 = np.dot(W2,a1) + b2
# a2 = activations.sigmoid(z2)
# print(f"\nSecond layer Activations\n{a2}")
# print(F"Shape of a2: {a2.shape}")

# # output layer
# W3 = np.random.rand(1,3) *0.0099
# b3 = np.random.rand(1,1)

# z3 = np.dot(W3,a2) + b3
# y_hat = activations.sigmoid(z3)
# print(f"\nThird layer Activations\n{y_hat}")
# print(F"Shape of y_hat: {y_hat.shape}")

# def logistic_classifier(vector):
#     return [1 if e > 0.5 else 0 for e in vector]

# print(f"Clasification: \n{logistic_classifier(y_hat[0])}")
    

neural_network = BasicNeuralNetwork(X=X, Y=Y)
neural_network.add_layers([(5,3),(3,5),(1,3)])
neural_network.forward_propagation()