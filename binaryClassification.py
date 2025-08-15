import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.datasets import make_circles # membuat sampe data
from sklearn.model_selection import train_test_split # membuat train split random
import pandas as pd
import requests
from pathlib import Path

# make 1000 samples
nSamples = 1000

X, y = make_circles(nSamples,
                    noise=0.03,
                    random_state=42)
print(len(X),len(y))
print(f"5 sampel pertama dari X: {X[:5]}")
print(f"5 sampel pertama dari y: {y[:5]}")
# value X stara denga value y yang berupa 1 dan 0,menandakan bahwa binary classification

# membuat dataframe dari lingkaran data
circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label": y})
print(circles.head(10))

# membuat graph
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu);
# plt.show()
## data yang di kerjakan diatas biasa disebut juga toy data set

## 1.1 cek bentuk input dan output
print(X.shape, y.shape) # y adalah label
## (1000,2) (1)

## 1.2 mengubah data ke tensor dan membuat train dan test split
# mengubah data ke tensor
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(f"5 sampel pertama dari X: {X[:5]}")
print(f"5 sampel pertama dari y: {y[:5]}")
print(X.dtype, y.dtype, type(X), type(y))

# split data
XTrain, XTest, yTrain, yTest = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=42)
# test size 0.2 artinya test data akan berisi 20% dari train data
# train data akan berisis 80% sisanya
# random_state adalah manual seed khusus sklearn dan manual seed tidak berfungsi untuk sklearn
print(len(XTrain), len(XTest), len(yTrain), len(yTest))


# 2. membuat model
# membuat model untuk mengklasifikasi titik merah dan biru

# setup:
# membuat agnostic mode
device = "cuda" if torch.cuda.is_available else "cpu"

# membuat model (membuat subclass nn.Module)
class CircleModeV0(nn.Module):
    def __init__(self):
        super(). __init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5) # in_features mengambil 2 data dari X dan membuatnya menjadi 5
        self.layer2 = nn.Linear(in_features=5 , out_features=1) # in_features di layer selanjutnya harus sesuai dengan out_features di layer sebelumnya, dan karna y hanyalah skalar maka out_features adalah 1

        # self.twoLinearLayers = nn.Sequential(
        #     nn.Linear(in_features= 2, out_features= 5),
        #     nn.Linear(in_features= 5, out_features= 1)
        # ) # penggunaan dengan sequential

    def forward(self, x):
        return self.layer2(self.layer1(x)) # x -> layer1 -> layer2 -> output
        # return self.twoLinearLayers(x) # penggunaan dengan sequential
# note:
# layer yang didefinisikan adalah jumlah input,karena value X berbentuk 2 value maka terdapat 2 layer
# out_feature adalah jumlah neuron yang akan mempebesar kemungkinan model akan training
# penggunaan model dengan nn.Sequential lebih mudah dibuat,namun penggunaan subclass module bisa mengatasi model yang lebih kompleks

# inisiasi model
model0 = CircleModeV0().to(device)
print(model0)
print(next(model0.parameters()).device)
# visualisasi hasil dari model :
#tensorflow playground

# replikasi model
model0 = nn.Sequential(
    nn.Linear(in_features= 2, out_features= 5),
    nn.Linear(in_features= 5, out_features=1)
).to(device)
print(model0)
# menghasilkan model yang sama dengan model sebelumnya

# visualisasi dari prediksi model
print(model0.state_dict())
with torch.inference_mode():
    untrainedPreds = model0(XTest.to(device))
print(f"len dari pred: {len(untrainedPreds)}, Shape: {untrainedPreds.shape}")
print(f"len dari test sampel: {len(XTest)}, Shape: {yTest.shape}")
print(f"\n10 pred pertama:\n {torch.round(untrainedPreds[:10])}")
print(f"\n10 label dari pred:\n {yTest[:10]}")

# mendefinisikan loss function dan optimizer
# loss func
# l1Loss / MAE(mean absolut error) berkerja untuk linear manun tidak untuk classification
# loss function yang digunakan untuk classification biasanya adalah:
# binary cross entrophy(binary)
# categorical cross entrophy(multiclass classification)

# setup loss func dengan torch.nn.BECWithLogitsLoss()
# pada loss function ini terdapat sigmoid function
lossFunc = torch.nn.BCEWithLogitsLoss()

# optimizer
# untuk optimizer yang biasa di gunakana adalah adam atau SGD
optimizer = torch.optim.SGD(params=model0.parameters(),
                            lr= 0.1)

# mengkalkulasi akurasi - menghtiung dari 100 sampel berapa persen model benar
def accuracyFN(yTrue, yPred):
    correct = torch.eq(yTrue.to(device), yPred.to(device)).sum().item()
    acc = (correct/len(yPred) * 100)
    return acc


# logits adalah output mentah yang dihasilkan oleh model(belum di train)
# raw logits -> prediction probabilities -> prediction labels

# raw logits bisa di ubah menjadi prediction probabilities dengan memberikannya ke activation function
# contoh:
# sigmpoid untuk binary classification
# softmax untuk multi-class classification

# lalu prediction probabilities di ubah menjadi prediction labels dengan cara:
# pembulatan / torch.round() untuk binary classification
# argmax() untuk multi-class classification

# melihat 5 output pertama pada forwad pass dengan test data
model0.eval()
with torch.inference_mode():
    yLogits = model0(XTest.to(device))[:5]
print(yLogits)

# menggunakan sigmoid activation function pada model logit ke predcitons prob
yPredProb = torch.sigmoid(yLogits)
print(yPredProb)
print(torch.round(yPredProb)) # dibulatkan

# untuk prediction prob values, kita perlu melakukan a range style pada meraka
# contoh:
# yPredProb >= 0.5, y=1 (class 1)
# yPredProb >= 0.5, y=1 (class 0)

# menemukan predicted label
yPreds = torch.round(yPredProb)

# dengan full code sebagai berikut (logtis -> pred prob -> pred labels)
yPredLabels = torch.round(torch.sigmoid(model0(XTest.to(device))[:5]))

# cek kesamaan
print(torch.eq(yPreds.squeeze(), yPredLabels.squeeze()))

# membuat training,testing loop dan evaluation loop
torch.cuda.manual_seed(42)
epochs = 100
# epochCount = []
# lossValue = []
# testLossValue = []

XTrain.to(device),yTrain.to(device)
XTest.to(device),yTrain.to(device)


for epoch in range(epochs):
    # forward pass
    yLogits = model0(XTrain.to(device)).squeeze()
    ypred = torch.round(torch.sigmoid(yLogits))

    # kalkulasi loss dan akurasi
    loss = lossFunc(yLogits,yTrain.to(device))
    # BCEWithLogits menggunakan logits untuk input
    # sedangkan BCELoss menggunakan pred prob untuk input menggunakan torch.sigmoid()
    Acc = accuracyFN(yTrue=yTrain,
                     yPred=ypred)

    # optimizer
    optimizer.zero_grad()

    # loss backward (backpropagation)
    loss.backward()

    # optimizer step (gradient descent)
    optimizer.step()

    # testing
    model0.eval()
    with torch.inference_mode():
        # forward pass
        testLogits = model0(XTest.to(device)).squeeze()
        testPred = torch.round(torch.sigmoid(testLogits))

        # kalkulasi test loss/accuracy
        testLoss = lossFunc(testLogits.to(device),
                            yTest.to(device))
        testAcc = accuracyFN(yTrue=yTest,
                             yPred=testPred)
        # print
        if epoch % 10 == 0:
            print(f"| Epoch: {epoch}/{epochs} | loss: {loss:.5f} Acc: {Acc:.2f}% | Test loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}% |")

# membuat visual predicitons
if Path("Helper_funtion.py").is_file():
    print("Already exists")
else:
    print("downloading")
    requests = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_function.py","wb") as f:
        f.write(requests.content)

from helper_function import plot_predictions, plot_decision_boundary

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model0, XTrain, yTrain)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model0, XTest, yTest)
# plt.show()

print(XTrain.squeeze().shape)
