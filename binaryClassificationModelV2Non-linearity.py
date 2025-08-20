import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.datasets import make_circles # membuat sampe data
from sklearn.model_selection import train_test_split # membuat train split random
import pandas as pd
import requests
from pathlib import Path


nSamples = 1000

X, y = make_circles(nSamples,
                    noise=0.03,
                    random_state=42)
# plt.scatter(X[:,0], X[:,1], c=y,cmap=plt.cm.RdYlBu)
# plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

Xtrain,Xtest,yTrain,yTest = train_test_split(X,
                                              y,
                                              test_size=0.2,
                                              random_state=42)

device = "cuda" if torch.cuda.is_available else "cpu"

class circleModelV2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        # gunakan non-linearity activation function
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model3 = circleModelV2().to(device)

# initiate loss fuinction and optimizer

lossFunction = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= model3.parameters(),
                            lr=0.1)
def accFun(yTrue,yPred):
    correct = torch.eq(yTrue, yPred).sum().item()
    acc = (correct/len(yPred) * 100)
    return acc

# train and test loop

torch.cuda.manual_seed(42)
epochs = 1000

Xtrain,yTrain = Xtrain.to(device), yTrain.to(device)
Xtest, yTest = Xtest.to(device),yTest.to(device)

for epoch in range (epochs):
    ylogits = model3(Xtrain).squeeze()
    yPred = torch.round(torch.sigmoid(ylogits))

    loss = lossFunction(ylogits,yTrain)
    acc = accFun(yTrue=yTrain,
                 yPred=yPred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model3.eval()
    with torch.inference_mode():
        testLogits = model3(Xtest).squeeze()
        testPred = torch.round(torch.sigmoid(testLogits))

        testLoss = lossFunction(testLogits,yTest)
        testAcc = accFun(yTrue=yTest,
                         yPred=testPred)

    if epoch % 100 == 0:
        print(f"| Epoch: {epoch+100}/{epochs} | Train Loss: {loss:.5f} | Train Acc: {acc:.2f}% | Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}%")

from helper_function import plot_decision_boundary

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model3, Xtrain, yTrain)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model3, Xtest,yTest)
plt.show()
