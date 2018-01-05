import numpy as np

class NeuralNetwork:
    def __init__(self, units_per_layer_array, learning_rate):
        """ Args: 
          - units_per_layer_array: an array containing the number of neurons in each layer
                                   its length is equal to the number of layers, including the 
                                   input layer and not including the output layer
                                   
          - learning_rate: eta value for the neural network
        """    
        # initializing member variables
        self.num_hidden_layers = len(units_per_layer_array) - 1
        
        self.weights_array = []
        self.a_array = [None] * (self.num_hidden_layers + 1)
        self.z_array = [None] * self.num_hidden_layers
        self.bias_array = []
        self.eta = learning_rate 
        
        # creating weights for hidden layers
        for u in range(self.num_hidden_layers):                    # number of layers excluding input layer
            num_inputs_curr_layer = units_per_layer_array[u]       # number of nodes in current layer
            num_outputs_curr_layer = units_per_layer_array[u+1]    # number of units in next layer
            
            # creating matrix of weights for hidden layer(s)
            hidden_weights = np.random.randn(num_outputs_curr_layer, 
                                             num_inputs_curr_layer)
            self.weights_array.append(hidden_weights)
            
            # creating matrix of biases for hidden layer(s)
            self.bias_array.append(np.random.randn(1, num_outputs_curr_layer))
        
        # creating matrix of weights for output
        num_inputs = units_per_layer_array[-1]   # getting number of neurons from last hidden layer
        num_outputs = 1
        
        output_weights = np.random.randn(num_outputs, num_inputs)
        self.weights_array.append(output_weights) # finish initializing weights
        self.bias_array.append(np.random.randn(1, 1))

    def predict(self, x):
        """ Args:
          - x: an array of length 'units_per_layer_array[0]`
               it is the input value(s) used to make a prediction
               should be a numpy array
        """
        current_layer = np.array([x])
        self.a_array[0] = current_layer
    
        # solving for h-values (hidden layer sums)
        for w in range(self.num_hidden_layers):
            z = np.dot(current_layer, self.weights_array[w].T) # summing weights and "inputs"

            z = z + self.bias_array[w]                       # add bias
            self.z_array[w] = z                              # appending sum of outputs (z)
            
            a = np.maximum(z, 0)                             # applying activation function
            self.a_array[w+1] = a                              # appending activation (a)
            
            current_layer = a
        
        # predicting y_hat without activation function
        y_hat_weights = self.weights_array[-1]               # getting last weights for y_hat
        y_hat = np.dot(current_layer, y_hat_weights.T)         # getting 1x1 matrix, which is y_hat
        y_hat = y_hat + self.bias_array[-1]  # add bias
        
        return y_hat[0][0] # returning y_hat value from 1x1 matrix
    
    
    def update(self, x, y):
        """Args:
          - x: an array of length 'units_per_layer_array[0]`
          - y: a float representing the output
        """
        
        ### notation
        # delta = error for the neuron
        # gradient = derivative of loss function wrt weight/bias
    
        ### compute prediction values via forward propogation
        y_hat = self.predict(x) # saves all the a values and z values for each layer
        y_error = 2*(y_hat - y)   # dL/dy_hat
        
        ### backpropogate error through layers
        delta_array = [None] * (self.num_hidden_layers + 1)
        gradient_array = [None] * (self.num_hidden_layers + 1)
        
        ## calculating deltas for all neurons
        # calculating neuron errors for layer before y_hat (column vector)
        last_layer_weights = self.weights_array[-1]
        delta_array[-1] = np.array([[y_error]])
        
        # calculating deltas for all layers except last one
        for l in range(self.num_hidden_layers - 1, -1, -1): # backpropogating
            # calculating left-hand side of Hadamard product
            weights = self.weights_array[l + 1]
            delta_next = delta_array[l + 1]
            left_hand_hadamard = np.dot(weights.T, delta_next)
            
            # calculating right-hand side of Hadamard product
            z = self.z_array[l].T
            g_prime = np.maximum(np.sign(z), 0)
            
            # calculating delta with Hadamard product
            delta_curr = np.multiply(left_hand_hadamard, g_prime)
            delta_array[l] = delta_curr
        
        ## calculating gradient for all weights
        for l in range(self.num_hidden_layers, -1, -1): 
            a = self.a_array[l]
            delta = delta_array[l]
            curr_layer_gradient = np.dot(delta, a)
            
            gradient_array[l] = curr_layer_gradient
            
        ## training weights using calculated gradients
        for l in range(len(self.weights_array)):
            delta = delta_array[l]
            gradient = gradient_array[l]
            
            self.weights_array[l] = self.weights_array[l] - (self.eta * gradient)
            #self.bias_array[l] = self.bias_array[l] - (self.eta * delta)
        
        new_y_hat = self.predict(x)
        return(y - new_y_hat)