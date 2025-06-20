import activations
import numpy as np


def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    # Add a small epsilon for numerical stability to prevent log(0)
    epsilon = 1e-8 # A very small positive number
    
    # Clip Y_hat values to be within [epsilon, 1 - epsilon] to avoid log(0) or log(1-1)
    Y_hat = np.maximum(epsilon, Y_hat)
    Y_hat = np.minimum(1 - epsilon, Y_hat)
    
    # Compute the binary cross-entropy loss
    logprobs = Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
    cost = - (1 / m) * np.sum(logprobs)
    return cost

class BasicNeuralNetwork():
    layers = []
    parameters = {}
    cache = {}
    grads = {}
    
    
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
    def initialize_params(self):
        """
        layer_dim are the dimensions of the neural network layer
        with the form (n,m) where:
            n: rows of the vector
            m: columns of the vector
            
        returns a tupple with the initialized weights and bias

        """
        for index, layer_dim in enumerate(self.layers):
            f,c = layer_dim
            self.parameters[f'W{index + 1}'] = np.random.rand(f,c) * 0.1
            self.parameters[f'b{index + 1}'] = np.random.rand(f,1) * 0.0000099
        
    
    def forward_propagation(self):
        W1 = self.parameters['W1']  # Shape: (n_h1, n_x) = (3, 3)
        b1 = self.parameters['b1']  # Shape: (n_h1, 1) = (3, 1)
        W2 = self.parameters['W2']  # Shape: (n_h2, n_h1)
        b2 = self.parameters['b2']  # Shape: (n_h2, 1)
        W3 = self.parameters['W3']  # Shape: (n_y, n_h2)
        b3 = self.parameters['b3']  # Shape: (n_y, 1)
        
        # Layer 1
        Z1 = np.dot(W1, self.X) + b1  # (3, 3) × (3, 5) + (3, 1) → (3, 5)
        A1 = activations.sigmoid(Z1)  # Shape: (3, 5)
        
        # Layer 2
        Z2 = np.dot(W2, A1) + b2  # (n_h2, 3) × (3, 5) + (n_h2, 1)
        A2 = activations.sigmoid(Z2)  # Shape: (n_h2, 5)
        
        # Output layer
        Z3 = np.dot(W3, A2) + b3  # (n_y, n_h2) × (n_h2, 5) + (n_y, 1)
        A3 = activations.sigmoid(Z3)  # Shape: (n_y, 5)
        
        self.cache = {'A1': A1, 'A2': A2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}
    
    def backward_propagation(self):
        """
        Perform backpropagation to compute gradients for a three-layer neural network.
        Assumes sigmoid activation and binary cross-entropy loss.
        """
        m = self.X.shape[1]
        
        # Retrieve parameters and cached activations
        W1 = self.parameters['W1']
        W2 = self.parameters['W2']
        W3 = self.parameters['W3']
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        A3 = self.cache['A3']
        
        # Output layer gradients (dZ3, dW3, db3)
        dZ3 = A3 - self.Y  # Gradient of loss w.r.t. Z3 (binary cross-entropy + sigmoid)
        dW3 = (1/m) * np.dot(dZ3, A2.T)  # Gradient of loss w.r.t. W3
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)  # Gradient of loss w.r.t. b3
        
        # Hidden layer 2 gradients (dZ2, dW2, db2)
        dZ2 = np.dot(W3.T, dZ3) * (A2 * (1 - A2))  # Backprop + sigmoid derivative
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer 1 gradients (dZ1, dW1, db1)
        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))  # Backprop + sigmoid derivative
        dW1 = (1/m) * np.dot(dZ1, self.X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Store gradients
        self.grads = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3
        }

    def update_parameters(self,learning_rate = 0.001):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        W3 = self.parameters['W3']
        b3 = self.parameters['b3']
        
        
        # Retrieve each gradient from the dictionary "grads"
        dW1 = self.grads['dW1']
        db1 = self.grads['db1']
        dW2 = self.grads['dW2']
        db2 = self.grads['db2']
        dW3 = self.grads['dW3']
        db3 = self.grads['db3']
    
        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W3 = W3 - learning_rate * dW3
        b3 = b3 - learning_rate * db3
    
        
        self.parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2, "W3": W3, "b3":b3}
    
    def train_model(self, num_iterations = 100000, learning_rate=0.001, print_cost = True):
        """

        """
        
        np.random.seed(3)
        
        

        self.initialize_params()
        # Loop (gradient descent)
        
        for i in range(0, num_iterations):
            
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            self.forward_propagation()
            Y_hat = self.cache['A3']
            # Cost function. Inputs: "A2, Y". Outputs: "cost".
            cost = compute_cost(Y_hat, self.Y)
    
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            self.backward_propagation()
    
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            self.update_parameters(learning_rate=learning_rate)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return self.parameters
        
            
            
    


X = np.array([[-110,-111,-12.5],[-2,0,-1.25], [-105,-0.22, -10.71],[-105,0.22, -10.71],[-0.105,-0.22, -0.71]])
XT = np.transpose(X)
Y = np.array([[1,1,0,1,1]]) # Note the extra set of brackets for 2D array

neural_network = BasicNeuralNetwork(X=XT, Y=Y)
neural_network.add_layers([(5,3),(3,5),(1,3)])
print(neural_network.train_model(num_iterations=500000, learning_rate=0.001))