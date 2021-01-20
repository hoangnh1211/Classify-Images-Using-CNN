import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt


from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

#Load data
#Using argument data
#Imput size = 224x224 because using Resnet pre-trained
transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_set = datasets.ImageFolder(root='./Animals-Dataset/train', transform=transform_train)
val_set = datasets.ImageFolder(root='./Animals-Dataset/val', transform=transform_val)

batch_size = 32

train_load = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_load = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

#Define
train_loss = []
val_loss = []
train_acc = []
val_acc = []

#Set requires_grad = False for pre-trained of Resnet, no update parameters in this
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#Define Transfer Learning net
def resnet_transfer(num_class):
    #Load Resnet18 have pre-trained
    resnet_transfer = models.resnet18(pretrained=True)

    #Set requires_grad = False in resnet.features
    set_parameter_requires_grad(resnet_transfer, feature_extracting=True)

    #Change FC layer with output = num_class
    num_ftr = resnet_transfer.fc.in_features
    resnet_transfer.fc = nn.Linear(num_ftr, num_class)

    return resnet_transfer

#Def training model
def Training_Model(model, epochs, parameters):
    #Using CrossEntropyLoss, optim SGD
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters, lr=0.01)

    # model = model.cuda() #for gpu

    for epoch in range(epochs):
        print("\nStarted epoch: ", epoch)
        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0
        model.train() #Set mode Train
        for i, (inputs, labels) in tqdm(enumerate(train_load, 0), total=len(train_load)):
            inputs = Variable(inputs)
            labels = Variable(labels)

            # inputs = inputs.cuda() #for gpu
            # labels = labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            iter_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1

        train_loss.append(iter_loss/iterations)
        train_acc.append((100 * correct / len(train_set)))

        #val_training
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval() #Set mode evaluation

        #No_grad on Val_set
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_load, 0):

                inputs = Variable(inputs)
                labels = Variable(labels)

                # inputs = inputs.cuda() #gpu
                # labels = labels.cuda()

                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            val_loss.append(loss/iterations)
            val_acc.append((100 * correct / len(val_set)))

        stop = time.time()
        pass
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}, Time: {}s'
            .format(epoch+1, epochs, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1],stop-start))

#Initial CNN
CNN = resnet_transfer(num_class=3)

#Summary NN
# print(CNN)

#Training model
epochs = 1
Training_Model(model=CNN, epochs=epochs, parameters=CNN.parameters())

#Save model
torch.save(CNN.state_dict(),'Animals_Model_Transferlearning_2.pth')

plt.plot(train_acc, label='Train_Acc')
plt.plot(val_acc, label='Val_Acc')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.axis('equal')
plt.legend(loc=7)