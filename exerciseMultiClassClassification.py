import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from helper_function import plot_decision_boundary
from torchmetrics import Accuracy

# non linear activation
class hyperTanh(nn.Module):
    def forward(self, x) :
        ex = torch.exp(x)
        enx = torch.exp(-x)
        return (ex - enx) / (ex + enx)

## dataset multiclass
# Code for creating a spiral dataset from CS231n
import numpy as np
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

print(X.shape,y.shape)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)
print(f"5 first value X: {X[:5]}")
print(f"5 first value y: {y[:5]}")
print(y.unique())

# trainsplit
XSpiralTrain,XSpiralTest,ySpiralTrain,ySpiralTest = train_test_split(X,
                                                                     y,
                                                                     test_size=0.2,
                                                                     random_state=42)

# membuat model

device = "cuda" if torch.cuda.is_available else "cpu"

class spiral(nn.Module):
    def __init__(self, input, ouput, neuron=10 ):
      """
      lagi lagi
      """
      super().__init__()
      self.nonLinear = nn.Sequential(
         nn.Linear(in_features=input, out_features=neuron),
         hyperTanh(),
         nn.Linear(in_features=neuron, out_features=neuron),
         hyperTanh(),
         nn.Linear(in_features=neuron, out_features=ouput),
      )
    def forward(self,x):
       return self.nonLinear(x)

modelEX = spiral(input=2,ouput=3).to(device)

# lossfn and optimizer

lossFn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=modelEX.parameters(),
                            lr= 0.1)

# train and test loop

XSpiralTrain,ySpiralTrain = XSpiralTrain.to(device),ySpiralTrain.to(device)
XSpiralTest,ySpiralTest = XSpiralTest.to(device),ySpiralTest.to(device)

torch.cuda.manual_seed(42)
epochs = 100

acc = Accuracy(task="multiclass", num_classes=3).to(device)

for epoch in range(epochs):
   yLogits = modelEX(XSpiralTrain)
   yPred = torch.softmax(yLogits, dim=1).argmax(dim=1)

   loss = lossFn(yLogits,ySpiralTrain)
   trainAcc = acc(yPred,ySpiralTrain)
   trainAcc = trainAcc.item()*100
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   modelEX.eval()
   with torch.inference_mode():
      testLogits = modelEX(XSpiralTest)
      testPred = torch.softmax(testLogits,dim=1).argmax(dim=1)

      testLoss = lossFn(testLogits,ySpiralTest)
      testAcc = acc(testPred, ySpiralTest)
      testAcc = testAcc.item() * 100

      if epoch % 10 == 0:
         print(f"| Epoch: {epoch+10}/{epochs} | Train Loss: {loss:.5f} | Train Acc: {trainAcc:.2f}% | Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}% |")

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(modelEX, XSpiralTrain,ySpiralTrain)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(modelEX,XSpiralTest,ySpiralTest)
plt.show()
