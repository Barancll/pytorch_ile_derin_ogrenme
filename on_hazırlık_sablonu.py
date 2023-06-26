##Matrixler;

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

array = np.random.randint(0,6,[1,2,3])
first_array = np.array(array)
print("Array Type: {}".format(type(first_array)))
print("Array Shape: {}".format(np.shape(first_array)))
print(first_array)


#Pytorch temelleri
'''
pytorch'da random array komutları;
np.ones() = torch.ones()
np.random.rand() = torch.rand()
'''
tensor = torch.Tensor(first_array)
print("Array Type{}".format(tensor.type))
print("Array Shape{}".format(tensor.type))
print(tensor)

#Allocation
print("Numpy {}\n".format(np.ones((2,3)))) #numpy ones
print(torch.ones((2,3))) #pytorch ones

print("Numpy {}\n".format(np.random.rand(2,3))) #numpy rand
print(torch.rand(2,3)) #pytorch rand


'''
Dönüştürme komutları
torch.from_numpy(): from numpy to tensor
numpy(): from tensor to numpy
'''
 #random numpy array
array = np.random.rand(2,2)
print(array)

#from numpy to tensor
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

#from tensor to numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy),from_tensor_to_numpy))


##Basic Math with Pytorch

#create tensor
tensor = torch.ones(3,3)
print("\n",tensor)

#resize
print("{} {}\n".format(tensor.view(9).shape,tensor.view(9)))

#Addition
print("Addition: {}\n".format(torch.add(tensor,tensor)))

#Subtraction
print("Subtraction: {}\n".format(tensor.sub(tensor)))

#Element wise multiplication
print("Element wise multiplication: {}\n".format(torch.mul(tensor,tensor)))

#element wise division
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

#Mean
tensor = torch.tensor([1,2,3,4,5], dtype=torch.float32)
print("Mean: {}".format(tensor.mean()))

#Standart deviation (std)
print("std: {}".format(tensor.std()))


##Veriable

#import variable from pytorch library
from torch.autograd import Variable

#define variable
var = Variable(torch.ones(3), requires_grad = True)
print(var)
 
#We have an equation that is y = x^2
array =[2,4]
tensor = torch.Tensor(array)
x = Variable(tensor, requires_grad = True)
y = x**2 
print("y = ",y)

#recap o equation o = 1/2*sum(y)
o = (1/2)*sum(y)
print("o = ",0)

#backward
o.backward() # calculates gradients

print("gradients: ", x.grad) 

##Linear Regression
#Veriyi oluşturma(Araç Fiyatları);

car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np = car_price_np.reshape(-1,1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))

#Numaralara göre araç çağıralım

number_of_car_sell_array = [7.5,7,7.6,6.5,6,5.5,4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1, 1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

#Print etmek yerine bu verileri çizdirelim
plt.scatter(car_prices_array, number_of_car_sell_array)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ vs Number of Car Sell")
plt.show()

##Linear Regression with Pytorch
import torch.nn as nn


#Create a module(Class)
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        #super function. it inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression, self).__init__()
        #Linear function.
        self.linear = nn.Linear(input_dim,output_dim)
        
    def forward(self,x):
        return self.linear(x)
    
#Define Model
input_dim =1
output_dim =1
model = LinearRegression(input_dim,output_dim) #input and output size are 1

#MSE
mse = nn.MSELoss()

#Optimization(find parameters that minimize error)
learning_rate = 0.02 #how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

#train model
loss_list = []
iteration_number = 1001 
for iteration in range(iteration_number):
    
    #optimization
    optimizer.zero_grad()
    
    #forward to get output
    results = model(car_price_tensor)
    
    #Calculate Loss
    loss = mse(results, number_of_car_sell_tensor)
    
    #backward propagation
    loss.backward()
    
    #Updating parameters
    optimizer.step()
    
    #Store loss
    loss_list.append(loss.data)
    
    #print loss
    if(iteration % 50 == 0) :
        print('epoch {}, loss {}'.format(iteration, loss.data))
        
plt.plot(range(iteration_number), loss_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")  
plt.show()      
        
#Predict our car price
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data", color = "red")
plt.scatter(car_prices_array,predicted,label = "predicted data", color = "blue")

#predict if car price is 10$, what will be the number of car sell
#predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10.predicted_10.data.label = "car price 10$", color="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title,("Original vs Predicted values")
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
    






























































