import pandas as pd

from activation_function import SignFunction

from perceptron import Perceptron

dataset = pd.read_csv('DataBase/treinamento.csv') 
X = dataset.iloc[:, 0:3].values
d = dataset.iloc[:, 3:].values
p = Perceptron(X, d, 0.01, SignFunction)

p.train()

print()
print(f'Initial W: {p.W0}')

print()
print('>>>>>>TESTE<<<<<<')
print(f'Input [-0,3665 0,0620 5,9891], Output {p.evaluate([-1,-0.3665, 0.0620, 5.9891])}') 
print(f'Input [-0,7842 1,1267 5,5912], Output {p.evaluate([-1, -0.7842, 1.1267, 5.5912])}') 
print(f'Input [0,3012 0,5611 5,8234], Output {p.evaluate([-1,  0.3012, 0.5611, 5.8234])}') 
print(f'Input [0,7757 1,0648 8,0677], Output {p.evaluate([-1,  0.7757, 1.0648, 8.0677])}') 
print(f'Input [0,1570 0,8028 6,3040], Output {p.evaluate([-1,  0.1570, 0.8028, 6.3040])}') 
print(f'Input [-0,7014 1,0316 3,6005], Output {p.evaluate([-1, -0.7014, 1.0316, 3.6005])}') 
print(f'Input [0,3748 0,1536 6,1537], Output {p.evaluate([-1,  0.3748, 0.1536, 6.1537])}') 
print(f'Input [-0,6920 0,9404 4,4058], Output {p.evaluate([-1, -0.6920, 0.9404, 4.4058])}') 
print(f'Input [-1,3970 0,7141 4,9263], Output {p.evaluate([-1, -1.3970, 0.7141, 4.9263])}') 
print(f'Input [-1,8842 -0,2805 1,2548], Output {p.evaluate([-1, -1.8842, -0.2805, 1.2548])}') 
print()