import numpy as np

class Perceptron:
    
    def __init__(self, input_values, output_values, learning_rate, activation_function):
        ones_column = np.ones((len(input_values), 1)) * -1
        self.input_values = np.append(ones_column, input_values, axis = 1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        self.W0 = self.W        
        
    def train(self):
        epochs = 1
        error = True
        
        while error:
            print(f'Epochs: {epochs}')
            error = False
            for x,d in zip(self.input_values, self.output_values):                
                u = np.dot(x, self.W)
                y = self.activation_function.g(u)
                print(f'Input: {x} Output: {y} Expeted: {d}')
                if(y != d):
                    print(f'Actual W: {self.W}')
                    self.W = self.W + self.learning_rate * (d - y) * x                    
                    error = True
                    print(f'New W: {self.W}')
            epochs += 1
            
            print('')
            print('')  
        print(f'Final W: {self.W}')        
                
    def evaluate(self, input_values):
        u = np.dot(input_values, self.W)
        return self.activation_function.g(u)
        
    
        
        

