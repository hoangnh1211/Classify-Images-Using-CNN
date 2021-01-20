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
import argparse
import torch.nn.functional as nnf

from PIL import Image
from torch.autograd import Variable

#args
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--path', default="", help='Path to image')

args = parser.parse_args()
# print(args.path)
# exit(0)

transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

CNN = resnet_transfer(num_class=3)
CNN.load_state_dict(torch.load('Animals_Model_Transferlearning.pth', map_location=torch.device('cpu')))

#Label encoder
label2id = {
    0: 'Cat',
    1: 'Dog',
    2: 'PanDa'
}

#Predict func
def predict(img_name, model, threshold):
    #Load img and transform
    image = cv2.imread(img_name)
    img = Image.fromarray(image)
    img = transform_val(img)
    img = img.view(1, 3, 224, 224)
    img = Variable(img)

    model.eval() #set eval mode

    #To Cuda
    model = model
    img = img

    output = model(img)

    predicted = torch.argmax(output)
    idx = predicted.item()
    prob = nnf.softmax(output, dim=1)
    top_p, top_class = prob.topk(1, dim = 1)

    proba = top_p.detach().numpy()[0].item()
    if(proba > threshold):
        p = label2id[idx]
    else: p = "Unknown"
    #Show origin img and predict
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    p = "Model predict: " + p 
    plt.title(p)
    plt.imshow(image)
    plt.show()

    return  p

print('Predict: ', predict(args.path, CNN, threshold=0.9))


