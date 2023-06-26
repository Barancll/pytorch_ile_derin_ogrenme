# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:45:21 2023

@author: bct
"""

##Pytorch with GPU

#İmport Library
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#Download
trainset = torchvision.datasets.CIFAR10(root = "./data",train = True, download = True, transform = transform)
testset = torchvision.datasets.CIFAR10(root = "./data",train = False, download = True, transform = transform)

#Data Load
batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size) 

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def imshow(img):
    """
    function to show images
    """
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()    

dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

# Define CNN
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y = wx + b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)



    
    def forward(self,x):
        # Max pooling over a (2,2) window
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




use_gpu = True


net = Net()

if use_gpu:
    device = torch.device("cuda:0")
    print("Device:", device)
    if torch.cuda.is_available():
        net = Net().to(device)
        print("GPU is available")
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name())

else:
    print("cpu")
    
    

# Loss Function and optimizer
error = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)





# Train the network
for epoch in range(2):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        if use_gpu and torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
print("Training is done")


# Test
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print("Predicted:", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))




