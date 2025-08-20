import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from helper_function import plot_decision_boundary
from torchmetrics import Accuracy

# membuat agnostic code

device = "cuda" if torch.cuda.is_available else 'cpu'

# membuat data set
nSamples = 1000

X,y = make_moons(n_samples=nSamples,
                 noise=0.1,
                 random_state=42)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X.shape, y.shape)

XTrain,XTest,yTrain,yTest = train_test_split(X,
                                             y,
                                             test_size=0.2,
                                             random_state=42)

# visualiasi data
# plt.scatter(x=X[:,0],
#             y=X[:,1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()

# membuat model

class moon(nn.Module):
    def __init__(self, input, output,nueron=20):
        """
        yah
        """
        super().__init__()
        self.nonLinear = nn.Sequential(
            nn.Linear(in_features=input,out_features=nueron),
            nn.ReLU(),
            nn.Linear(in_features=nueron, out_features=nueron),
            nn.ReLU(),
            nn.Linear(in_features=nueron, out_features=nueron),
            nn.ReLU(),
            nn.Linear(in_features=nueron, out_features=nueron),
            nn.ReLU(),
            nn.Linear(in_features=nueron, out_features=output),
        )
    def forward(self, x):
        return self.nonLinear(x)

modelEx = moon(input=2,
               output=1,
               nueron=20).to(device)

# setup loss function and optimizer
lossfn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=modelEx.parameters(),
                            lr=0.1)
acc = Accuracy(task="binary").to(device)
# Training and Testing loop

XTrain,yTrain = XTrain.to(device),yTrain.to(device)
XTest,yTest = XTest.to(device),yTest.to(device)

torch.cuda.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    yLogits = modelEx(XTrain).squeeze()
    yPred = torch.round(torch.sigmoid(yLogits))

    trainLoss = lossfn(yLogits, yTrain)
    trainAcc = acc(yPred,yTrain)
    trainAcc = trainAcc.item() * 100
    optimizer.zero_grad()
    trainLoss.backward()
    optimizer.step()

    modelEx.eval()
    with torch.inference_mode():
        testLogits = modelEx(XTest).squeeze()
        testPred = torch.round(torch.sigmoid(testLogits))

        testLoss = lossfn(testLogits,yTest)
        testAcc = acc(testPred,yTest)
        testAcc = testAcc.item() * 100

        if epoch % 10 == 0:
            print(f"| Epoch: {epoch+10}/{epochs} | Train Loss: {trainLoss:.5f} | Train Acc: {trainAcc:.2f}% | Test Loss: {testLoss:.5f} | Test Acc: {testAcc:.2f}% |")

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(modelEx, XTrain, yTrain)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(modelEx, XTest, yTest)
plt.show()


