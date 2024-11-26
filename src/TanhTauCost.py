#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes):
        # Initializing List to Create Epoch vs. Loss Graph
        self.EpochList = [] 
        self.LossList = []
        

        self.num_layers = len(sizes)  #Number of Elements in array "sizes" gives number of layers
        self.sizes = sizes # Making sizes into a member of the class
        self.biases = [np.array([[-10], [5], [0]]), np.array([[5]])] # Iniializing the biases

        # 3D array [i][j][k]; i denotes the layer, j denotes the neuron number, k denotes that the value is needed as an int not array(each bias is in an array with only one element)

        self.weights = [np.array([[0.5], [0.5], [0.5]]), np.array([[ 0.5,0.5, 0.5]])] #Initializing the weights
        # 

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = np.tanh(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if test_data: n_test = len(test_data) # 
        n = len(training_data)
        for j in range(epochs):
            self.EpochList.append(j)
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2-time1))
            
            #print(self.biases)
            #print(self.weights)


    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] # .shape prints number of elements in each dimesnion of array i.e ([[1, 2, 3, 4], [5, 6, 7, 8]]).shape => (2,4)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #print(x)
            #print(y)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = np.tanh(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            tanh_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = tanh_prime(z)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        test_results = [tuple(((self.feedforward(x)[0][0]), y))
                        for (x, y) in test_data]
        print("Test Results: ", test_results)
        sum=0
        Cost =0
        for i in range(len(test_results)):
            if (taucost(test_results[i][0],test_results[i][1]) < 0.1):
                sum+=1 
            else:
                print(taucost(test_results[i][0],test_results[i][1]))
        Cost+=taucost(test_results[i][0],test_results[i][1])
        print("Cost:", Cost)
        self.LossList.append(Cost)
        
        return sum

    def cost_derivative(self, output_activations, y):
        return (2*(output_activations-y)/(1-(output_activations)**2))

    def graph_epoch_vs_loss(self):
        plt.plot(self.EpochList, self.LossList)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch vs. Loss for Hyperbolic Tangent Activation and Tau Cost")
        plt.show()

#### Miscellaneous functions

def tanh_prime(z):
    """Derivative of Hyperbolic Tangent"""
    return (1 - (np.tanh(z))**2)
def taucost(a, y):
    return (-1*((1-y)*np.log(1-a)+(1+y)*np.log(1+a))+2*np.log(2))


"""Testing the Sigmoid with Quadratic Cost"""
TrainingData = []
for i in range(100):
    TrainingData.append((i/100, 1- (i/100)))
TestData = [(0.25, 0.75),
            (0.45, 0.55),
            (0.33,0.67)]

TestNetwork = Network([1,3,1])


TestNetwork.SGD(TrainingData, 1000, 1, 0.001, TestData)
print(TestNetwork.feedforward((0.4)))
TestNetwork.graph_epoch_vs_loss()


# Result: 
"""Average Distance between Actual and Output"""
sum = 0
for a,y in TestData:

    sum += np.absolute(TestNetwork.feedforward(a)-y)

sum = sum/len(TestData)
print(sum)

#sum = 0.02925639