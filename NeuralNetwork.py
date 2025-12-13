#for mathematical operations
import numpy as np 
#for loading the breast cancer dataset 
from sklearn.datasets import load_breast_cancer
#to split the dataset into training and testing data
from sklearn.model_selection import train_test_split
#we import this to get all features in one range of numbers
from sklearn.preprocessing import StandardScaler
#we need this for our neural network's evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Functions for neural network

#sigmoid function for output
#Sigmoid activation function: 1 / (1 + e^-z)
def sigmoid(z):
    return 1 / (1+ np.exp(-z))

#relu function for our hidden layer
#ReLU activation function: max(0, z)
def relu(z):
    return np.maximum(0,z)

#derivative of relu needed in backpropagagation
def der_relu(z):
    return (z>0).astype(float)

#Class for neural network
class NeuralNetwork:
    #constructor for storing hyperparameters
    def __init__(self,input_size,hidden_size,output_size,learning_rate = 0.01):
        # FIX: Use the variable passed in, don't hardcode 0.01
        self.learning_rate = learning_rate
        #intializing weights with small random values
        #weights shape is (layer_neurons,input_connections)
        self.params = {
            'W1': np.random.randn(hidden_size, input_size) * 0.01,
            'B1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * 0.01,
            'B2': np.zeros((output_size, 1))
        }

    #defining forwardpropagation function for predicting the outputs
    def forward_propagation(self,X):
        #here we take inputs intialized randomly froms self.params

        #layer 1 hidden layer 
        # z = w*x+b
        Z1 = np.dot(self.params['W1'],X) + self.params['B1']
        #activation function for hidden layer
        A1 = relu(Z1)

        #layer 2 output layer
        # z' = w'*a1+b2
        Z2 = np.dot(self.params['W2'], A1) + self.params['B2']
        #activation function for output layer is sigmoid because we should classify it
        A2 = sigmoid(Z2)

        # Store for backprop
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache
    
    #loss function to find error
    def loss(self,A2,Y):
        #it gives how many no of outputs
        m = Y.shape[1]
        #we use epilson to make sure our computer never caluclates log(0) under any conditions
        epsilon = 1e-15
        cost = - (1/m) * np.sum(Y * np.log(A2 + epsilon) + (1-Y) * np.log(1-A2 + epsilon))
        #squeeze function gets out value from array
        return np.squeeze(cost)
    
    #defining function for backwordpropagation for learning and updating parameters for better predictions
    # FIX: Corrected spelling from 'backword' to 'backward'
    def backward_propagation(self,X,Y,cache):
        m = Y.shape[1]

        # Retrieve cache
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        W2 = self.params['W2']

        #gradients for output layer
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        #axis = 1 means horizonatal and rows and keepdims=true means we are protecting shape because numpy will update the shape
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        #gradients for hidden layers
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * der_relu(Z1) # Element-wise multiplication
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # FIX: Keys in gradients must match logic, but the update function decides what keys to look for in self.params
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradients
    
    #function for updating the parameters
    def update_parameters(self,gradients):
        #updating the weights and biases of layers from gradients
        # FIX: Changed 'b1'/'b2' to 'B1'/'B2' to match your __init__ keys
        self.params['W1'] -= self.learning_rate * gradients['dW1']
        self.params['B1'] -= self.learning_rate * gradients['db1']
        self.params['W2'] -= self.learning_rate * gradients['dW2']
        self.params['B2'] -= self.learning_rate * gradients['db2']
    
    #function for training the neural network
    #print_cost =  True will help us to see the training results for each epochs
    def train(self , X, Y,epochs = 1000, print_cost = True):
        #epochs means how many times we are training /updating  parameters
        for i in range(epochs):
            # 1. Forward Prop
            A2, cache = self.forward_propagation(X)
            
            # 2. Compute Loss
            cost = self.loss(A2, Y)
            
            # 3. Backward Prop
            # FIX: Updated function call name to match corrected definition
            gradients = self.backward_propagation(X, Y, cache)
            
            # 4. Update Weights
            self.update_parameters(gradients)

            #printing xost after every 100 epochs
            if print_cost and i % 100 == 0:
                print(f"Cost after epoch {i}: {cost:.4f}")
    
    #function for predicting the output
    def predict(self, X):
        #runs the neural network logic
        A2, _ = self.forward_propagation(X)
        #applies a threshold value for classification
        predictions = (A2 > 0.5).astype(int)
        return predictions

#loading data
data = load_breast_cancer()
#extracts the input data
X_orig = data.data
#this extracts the labels 0/1 outputs
y_orig = data.target.reshape(1, -1)

#scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)

#splitting the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_orig.T, test_size=0.2, random_state=42)

#transposing our inputs and outputs so our features*examples matches 
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

#shape of training and testing data
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

#creating a instance of neuralnetwork to get the prediction
model = NeuralNetwork(input_size=30, hidden_size=16, output_size=1, learning_rate=0.01)
print("\n -> Starting Training...")
model.train(X_train, y_train, epochs=1000)

#prediction
print("\n--- Evaluation ---")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#Evaluation metrics
train_acc = accuracy_score(y_train.flatten(), y_pred_train.flatten())
test_acc = accuracy_score(y_test.flatten(), y_pred_test.flatten())

#here we multiply accuracy values 100 beacuse it is in 0-1 range and we need in 1-100 range
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

#output

sample = X_test[:, 5].reshape(-1, 1)
print(f"Input: {sample.flatten()}")
print(f"Actual: {y_test[0, 5]}")
print(f"Predicted: {model.predict(sample)[0, 0]}")