import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veriyi Oku-Yükle
train = pd.read_csv("C:/Users/bct/Desktop/pytorch_udemy/Logistic_regression/train.csv", dtype=np.float32)

# Veriyi özellikler (pikseller) ve etiketler (0-9 arası sayılar) olarak ayır
targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != "label"].values / 255  # normalizasyon

# Eğitim ve test verilerini ayır (%80-%20)
features_train, features_test, targets_train, targets_test = train_test_split(
    features_numpy, targets_numpy, test_size=0.2, random_state=42
)

# Eğitim seti için özellikler ve etiketleri tensörlere dönüştür
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)  # Veri tipini LongTensor olarak ayarla

# Test seti için özellikler ve etiketleri tensörlere dönüştür
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # Veri tipini LongTensor olarak ayarla

# Batch boyutu, epoch ve iterasyon sayısı
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# PyTorch eğitim ve test veri setleri
train = TensorDataset(featuresTrain, targetsTrain)
test = TensorDataset(featuresTest, targetsTest)

# DataLoader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

# Veri setindeki bir resmi görselleştir
plt.imshow(features_numpy[10].reshape(28, 28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()


# Lojistik Regresyon Modeli Oluştur
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # Lineer kısım
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# Model sınıfını örnekleyin
input_dim = 28 * 28  # Görüntü boyutu px * px
output_dim = 10  # Etiketler 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

# Lojistik regresyon modelini oluştur
model = LogisticRegressionModel(input_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Modeli Eğitme
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Değişkenleri tanımla
        train = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Gradyanları sıfırla
        optimizer.zero_grad()

        # İleri yayılım
        outputs = model(train)

        # Softmax ve Cross Entropy Loss hesapla
        loss = error(outputs, labels)

        # Gradyanları hesapla
        loss.backward()

        # Parametreleri güncelle
        optimizer.step()

        count += 1

        # Tahmin et
        if count % 50 == 0:
            # Doğruluğu hesapla
            correct = 0
            total = 0

            # Test veri seti üzerinde tahmin yap
            for images, labels in test_loader:
                test = Variable(images.view(-1, 28 * 28))

                # İleri yayılım
                outputs = model(test)

                # En yüksek değeri kullanarak tahminleri al
                predicted = torch.max(outputs.data, 1)[1]

                # Toplam etiket sayısı
                total += labels.size(0)

                # Doğru tahmin edilen sayısı
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # Loss ve iterasyonu kaydet
            loss_list.append(loss.item())
            iteration_list.append(count)

            if count % 500 == 0:
                # Loss ve doğruluk değerlerini yazdır
                print('İterasyon: {} Loss: {} Doğruluk: {}%'.format(count, loss.item(), accuracy))

# Görselleştirme
plt.plot(iteration_list, loss_list)
plt.xlabel("iterasyon sayısı")
plt.ylabel("loss")
plt.title("Lojistik Regresyon: Loss vs İterasyon")
plt.show()


## Yapay Sinir Ağı (ANN)
import torch
import torch.nn as nn
from torch.autograd import Variable

# ANN Modeli Oluşturma
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()

        # Lineer fonksiyon 1: 784-150
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Doğrusal olmayanlık 1
        self.relu1 = nn.ReLU()

        # Lineer fonksiyon 2: 150-150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Doğrusal olmayanlık 2
        self.tanh2 = nn.Tanh()

        # Lineer fonksiyon 3: 150-150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Doğrusal olmayanlık 3
        self.elu3 = nn.ELU()

        # Lineer fonksiyon 4: 150-10
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Lineer fonksiyon 1
        out = self.fc1(x)
        # Doğrusal olmayanlık 1
        out = self.relu1(out)

        # Lineer fonksiyon 2
        out = self.fc2(out)
        # Doğrusal olmayanlık 2
        out = self.tanh2(out)

        # Lineer fonksiyon 3
        out = self.fc3(out)
        # Doğrusal olmayanlık 3
        out = self.elu3(out)

        # Lineer fonksiyon 4
        out = self.fc4(out)

        return out


# Model sınıfını örnekleyin
input_dim = 28 * 28  # Görüntü boyutu px * px
hidden_dim = 150  # Gizli katman boyutu
output_dim = 10  # Etiketler 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

# Yapay Sinir Ağı (ANN) modelini oluştur
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Modeli Eğitme
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Değişkenleri tanımla
        train = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Gradyanları sıfırla
        optimizer.zero_grad()

        # İleri yayılım
        outputs = model(train)

        # Softmax ve Cross Entropy Loss hesapla
        loss = error(outputs, labels)

        # Gradyanları hesapla
        loss.backward()

        # Parametreleri güncelle
        optimizer.step()

        count += 1

        # Tahmin et
        if count % 50 == 0:
            # Doğruluğu hesapla
            correct = 0
            total = 0

            # Test veri seti üzerinde tahmin yap
            for images, labels in test_loader:
                test = Variable(images.view(-1, 28 * 28))

                # İleri yayılım
                outputs = model(test)

                # En yüksek değeri kullanarak tahminleri al
                predicted = torch.max(outputs.data, 1)[1]

                # Toplam etiket sayısı
                total += labels.size(0)

                # Doğru tahmin edilen sayısı
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # Loss ve iterasyonu kaydet
            loss_list.append(loss.item())
            iteration_list.append(count)

            if count % 500 == 0:
                # Loss ve doğruluk değerlerini yazdır
                print('İterasyon: {} Loss: {} Doğruluk: {}%'.format(count, loss.item(), accuracy))

# Görselleştirme
plt.plot(iteration_list, loss_list)
plt.xlabel("iterasyon sayısı")
plt.ylabel("loss")
plt.title("Yapay Sinir Ağı (ANN): Loss vs İterasyon")
plt.show()



##CNN
 # Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

# batch_size, epoch and iteration
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
# Create CNN
model = CNNModel()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()