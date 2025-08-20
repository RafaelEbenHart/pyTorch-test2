import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.datasets import make_blobs # membuat sampel data
from sklearn.model_selection import train_test_split # membuat train split random
import pandas as pd
import requests
from pathlib import Path
from helper_function import plot_decision_boundary
from torchmetrics import Accuracy # setup accuracy dan metrics lainnya

# membuat dataset untuk multi-class classification
nClasses = 4
nFeatures = 2
randomSeed = 42
nSampel = 1000

# 1. Membuat multi class data
Xblob, yBlob = make_blobs(n_samples = nSampel,
                          n_features= nFeatures,
                          centers=nClasses,
                          cluster_std= 1.5, # membuat class menjadi sedikit random
                          random_state=randomSeed)

# 2. Mengubah data ke tensor
Xblob = torch.from_numpy(Xblob).type(torch.float)
yBlob = torch.from_numpy(yBlob).type(torch.LongTensor) # mengubah data y menjadi longTensor
print(f"5 sampel pertama dari Xblob: {Xblob[:5]}")
print(f"5 sampel pertama dari yBlob: {yBlob[:5]}")

# 3. TrainSplit
XblobTrain,XblobTest,yBlobTrain,yBlobTest = train_test_split(Xblob,
                                                              yBlob,
                                                              test_size=0.2,
                                                              random_state=randomSeed)
## note: XX,YY

# 4. Visualisasi data
plt.figure(figsize=(10,5))
plt.scatter(Xblob[:,0] , Xblob[:,1], c=yBlob, cmap=plt.cm.RdYlBu)
# plt.show()

# 5. Membuat model

device = "cuda" if torch.cuda.is_available else "cpu"

class blobModel(nn.Module):
    def __init__(self, inputFeatures, outputFeatures, hiddenUnits=8):
        """
        Initialize multi-class classification

        Args:
            inputFeatures (int): Number of input features to them model
            outputFeatures (int): Number of output features(number of output classes)
            hiddeUnits (int): Number of hidden units between layers,default 8

        Returns:

        Example:
        """
        super().__init__()
        self.linearLayerStack = nn.Sequential(
            nn.Linear(in_features=inputFeatures, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=outputFeatures),
        )
    def forward(self, x):
        return self.linearLayerStack(x)

model4 = blobModel(inputFeatures=2,
                   outputFeatures=4,
                   hiddenUnits=8).to(device)

def accuracyFN(yTrue, yPred):
    correct = torch.eq(yTrue.to(device), yPred.to(device)).sum().item()
    acc = (correct/len(yPred) * 100)
    return acc
# 6. Setup Loss Function dan optimizer

lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model4.parameters(),
                            lr= 0.1)

# 7. Setup Training and Testing loop

XblobTrain,yBlobTrain = XblobTrain.to(device), yBlobTrain.to(device)
XblobTest, yBlobTest = XblobTest.to(device) , yBlobTest.to(device)

torch.cuda.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model4.train()
    yLogits = model4(XblobTrain)
    yPred = torch.softmax(yLogits, dim=1).argmax(dim=1)

    loss = lossFunction(yLogits, yBlobTrain)
    acc = accuracyFN(yBlobTrain, yPred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model4.eval()
    with torch.inference_mode():
        testLogits = model4(XblobTest)
        testPred = torch.softmax(testLogits, dim=1).argmax(dim=1)
        testLoss = lossFunction(testLogits,yBlobTest)
        testAcc = accuracyFN(yBlobTest,testPred)
    if epoch % 10 == 0:
        print(f"| Epoch: {epoch+10}/{epochs} | Train Loss: {loss:.5f} | Train Acc: {acc:.2f}% | Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}% |")


model4.eval()
with torch.inference_mode():
    yLogits = model4(XblobTest)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model4, XblobTrain, yBlobTrain)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model4, XblobTest, yBlobTest)
plt.show()

## Beberapa hal yang bisa di evaluasi dari classification model
# 1. Accuracy - dari 100 sampel,berapa persen model benar torchmetrics.Accuracy()
# 2. Precision - torchmetrics.Precision()
# 3. Recall - torchmetrics.Recall()
# 4. F1-Score - torchmetrics.F1Score()
# 5. Confusion matrix -torchmetrics.ConfusionMatrix()
# 6. Classification report -sklearn.metrics.classification_report
## torchMetrics

## setup metric
torchMetricAcc = Accuracy(task="multiclass", num_classes=4).to(device)
## kalkulasi akurasi
print(torchMetricAcc(testPred,yBlobTest))
## note:
# pastikan targetkan device karena torchMetrics menggunakan cpu
