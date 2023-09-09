
import torch
from torch import nn
from torchvision.transforms import Compose, PILToTensor, ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

PREFERRED_DEVICE='cuda'

    
class ConvolutionalModel(nn.Module):
    def __init__(self,activation_fn=nn.LeakyReLU):
        super().__init__()
        self.to(PREFERRED_DEVICE)
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            activation_fn(3e-2),
            nn.Conv2d(32, 64, (3,3)), 
            activation_fn(1.8e-2),
            nn.Conv2d(64, 64, (3,3)),
            activation_fn(),
            nn.Flatten(), 
            nn.Linear(64*22*22, 10),
        )
    

    def forward(self, x): 
        x = x.to(PREFERRED_DEVICE)
        return self.model(x)

class LinearNetwork2(nn.Module):
    def __init__(self,activation_function=nn.LeakyReLU):
        super().__init__()
        self.to(PREFERRED_DEVICE)
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            activation_function(),
            nn.Linear(512,100),
            activation_function(),
            nn.Linear(100,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(PREFERRED_DEVICE)
        return self.model(x)


class LinearNetwork(nn.Module):
    def __init__(self,activation_function=nn.LeakyReLU):
        super().__init__()
        self.to(PREFERRED_DEVICE)
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 686),
            activation_function(),
            nn.Linear(686 , 512),
            activation_function(),
            nn.Linear(512,480),
            activation_function(),
            nn.Linear(480,256),
            activation_function(),
            nn.Linear(256,128),
            activation_function(),
            nn.Linear(128,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(PREFERRED_DEVICE)
        return self.model(x)


model = ConvolutionalModel()
epochs=10000
learning_rate=6e-2
batch_size=64


loss_fn= nn.CrossEntropyLoss()
optimizer=SGD(model.parameters(),lr=learning_rate)


DATASET_TEST=datasets.MNIST(root='mnist', train=False, transform=ToTensor(), download=True)
DATASET_TRAIN=datasets.MNIST(root='mnist', train=True, transform=ToTensor(), download=True)
DATALOADER_TEST=DataLoader(DATASET_TEST,batch_size=batch_size)
DATALOADER_TRAIN=DataLoader(DATASET_TRAIN,batch_size=batch_size)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    model.to(PREFERRED_DEVICE)
    for batch, (X, y) in enumerate(dataloader):
        X=X.to(PREFERRED_DEVICE)
        y=y.to(PREFERRED_DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            


def test_loop(dataloader, model, loss_fn):
    model.eval()
    model.to(PREFERRED_DEVICE)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(PREFERRED_DEVICE)
            y=y.to(PREFERRED_DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



for epoch in range(1,epochs):
    print(f'Epoch {epoch} {20*"-"}')
    train_loop(DATALOADER_TRAIN,model,loss_fn,optimizer)
    test_loop(DATALOADER_TEST,model,loss_fn)

print('\nExit\n')