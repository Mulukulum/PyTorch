import subprocess
import sys
import logging
import random
import os
import typing

RUNNING_SCRIPT_FROM = sys.argv[0]
DIRECTORY=os.path.dirname(RUNNING_SCRIPT_FROM)
LOGGING_LEVEL= logging.INFO
PREFERRED_DEVICE= 'cpu'

try:
    open('settings.txt','x')
except FileExistsError:
    with open('settings.txt','r') as file:
        USE_CUDA_WHEN_AVAILABLE=0 if file.read() in ('','0') else 1
    with open('settings.txt','w') as file:
        file.write(f'{USE_CUDA_WHEN_AVAILABLE}')
else:
    USE_CUDA_WHEN_AVAILABLE=0

logging.basicConfig(filename='NeuralNetwork.log',level=LOGGING_LEVEL)

def install_package(package_name: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

try:
    from PIL import Image as PillowImage
    import PyQt5
    from PyQt5 import QtWidgets
    from PyQt5 import QtCore
    from PyQt5 import QtGui
    
except:
    try:
        install_package('pillow')
        install_package('sip')
        install_package('PyQt5')
        install_package('PyQt5-sip')
    except:
        logging.CRITICAL('Something went wrong in the installation step for the packages')
        input('Package installation failed, press enter to continue')
        sys.exit()

try:
    import torch
    from torch import nn
    from torchvision.transforms import ToTensor
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torch.optim import SGD
except:
    print('Please have PyTorch installed on your system')
    input('Press Enter to Continue')
    sys.exit()

from ui_Main import Ui_MainWindow
from ui_Settings import Ui_SettingsWidget

#Main Functions
def get_cuda_information():
    device_count=torch.cuda.device_count()
    return dict([(i,torch.cuda.get_device_name()) for i in range(device_count)])

def save_model_state_dict(model_state_dict: dict ,filepath: str ) -> None :
    torch.save(model_state_dict,filepath)
    logging.info(f'File Saved to {filepath}')

def get_model_state_dict(filepath: str, current_device: str ) -> dict : 
    return torch.load(filepath, map_location=torch.device(current_device))
    
class ConvolutionalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.mdltype='Conv'
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.LeakyReLU(),
            nn.Flatten(), 
            nn.Linear(64*22*22, 10)
        )
    
    def forward(self, x): 
        return self.model(x)

class LinearNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.mdltype='Linear'
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512,100),
            nn.LeakyReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class SettingsWidget(QtWidgets.QWidget):

    def __init__(self, frame):
        super().__init__(parent=frame)
        self.ui=Ui_SettingsWidget()
        self.ui.setupUi(self)
        global USE_CUDA_WHEN_AVAILABLE
        if USE_CUDA_WHEN_AVAILABLE : self.ui.CudaBox.setChecked(True)
        else : self.ui.CudaBox.setChecked(False) 
        self.ui.CudaBox.stateChanged.connect(lambda : self.ChangeCudaSetting())
    
    def ChangeCudaSetting(self):
        global USE_CUDA_WHEN_AVAILABLE
        if USE_CUDA_WHEN_AVAILABLE:
            USE_CUDA_WHEN_AVAILABLE=0
        else:
            USE_CUDA_WHEN_AVAILABLE=1
        with open('settings.txt','w') as file:
            file.write(f'{USE_CUDA_WHEN_AVAILABLE}')

class LoadTestWidget(QtWidgets.QWidget):

    def __init__(self,frame):
        super().__init__(frame)

class TrainSaveWidget(QtWidgets.QWidget):

    def __init__(self,frame):
        super().__init__(frame)

class PTAppMainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(960,540)
        self.widgetFrame=QtWidgets.QFrame()
        self.ui.SettingsButton.clicked.connect(lambda : self.showSettingsWidget())
        self.ui.LoadTestButton.clicked.connect(lambda: self.showLoadTestWidget())
        self.ui.TrainSaveButton.clicked.connect(lambda: self.showTrainSaveWidget())
    
    def showSettingsWidget(self):
        self.widgetFrame.deleteLater()
        settings_widget=SettingsWidget(self.ui.MainWindowFrame)
        self.ui.VLayoutForFocusWidget.addWidget(settings_widget)
        self.widgetFrame=settings_widget
    
    def showLoadTestWidget(self):
        self.widgetFrame.deleteLater()
        loadtest_widget=LoadTestWidget(self.ui.MainWindowFrame)
        self.ui.VLayoutForFocusWidget.addWidget(loadtest_widget)
        self.widgetFrame=loadtest_widget
    
    def showTrainSaveWidget(self):
        self.widgetFrame.deleteLater()
        trainsave_widget=TrainSaveWidget(self.ui.MainWindowFrame)
        self.ui.VLayoutForFocusWidget.addWidget(trainsave_widget)
        self.widgetFrame=trainsave_widget
    

class ModelTrainer():

    def __init__(self, model: torch.nn.Module,loss_fn=None,
                 optimizer=None, number_of_epochs: int=10, 
                 learning_rate: float=0.005, batch_size: int=16,
                 status_report_after: int = 1  ):
        
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        if optimizer is None:
            self.optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        DATA=datasets.MNIST(root='mnist', train=True, transform=ToTensor(), download=True)
        DATASET=DataLoader(DATA, batch_size=batch_size)
        model.train()

        self.epoch_status_messages = ['',]
        for epoch in range(1, number_of_epochs+1) : 
            for batch in DATASET:
                input_batch , actual_values = batch
                input_batch = input_batch.to(PREFERRED_DEVICE)
                actual_values = actual_values.to(PREFERRED_DEVICE)
                guess = model(input_batch)
                loss = self.loss_fn(guess, actual_values)
                
                #backprop step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch%status_report_after ==0 :
                    self.epoch_status_messages.append(f"Epoch:{epoch} loss = {loss.item()}\n")

        #model has finished training at this point

class ModelTester():

    def __init__(self, model: nn.Module, loss_fn=nn.CrossEntropyLoss(), batch_size: int=1 ) -> tuple:        
        DATA=datasets.MNIST(root='mnist', train=True, transform=ToTensor(), download=True)
        DATASET=DataLoader(DATA,batch_size=batch_size)
        model.eval()
        size = len(DATASET.dataset)
        num_batches = len(DATASET)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, actual_value in DATASET:
                guess = model(inputs)
                test_loss += loss_fn(guess, actual_value).item()
                correct += (guess.argmax(1) == actual_value).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        self.average_loss=test_loss
        self.correct_percentage=correct*100
        self.results = f"Test Error: \n Accuracy: {self.correct_percentage:>0.1f}%, Avg loss: {self.average_loss:>8f} \n"
        return (self.correct_percentage,self.average_loss,self.results)

class ModelTester_Manual():
    
    def __init__(self, model) : 
        if 'MNIST Dataset JPG format' not in os.listdir(DIRECTORY):
            logging.CRITICAL('DATASET NOT PRESENT. UNZIP THE FOLDER INSIDE TO THE DIRECTORY OF THE SCRIPT WITH THE NAME "MNIST Dataset JPG format" ')
            sys.exit()
        self.model=model
        self.number=random.choice('0123456789')
        self.random_file=random.choice(os.listdir(DIRECTORY+f'//MNIST Dataset JPG format//MNIST - JPG - testing//{self.number}//'))
        self.imgpath=DIRECTORY+f'//MNIST Dataset JPG format//MNIST - JPG - testing//{self.number}//{self.random_file}'
        self.testing=False
        self.guesses=0
        self.correct_guesses=0
        self.accuracy=0.00

    def update_file(self):
        self.number=random.choice('0123456789')
        self.random_file=random.choice(os.listdir(DIRECTORY+f'//MNIST Dataset JPG format//MNIST - JPG - testing//{self.number}//'))
        self.imgpath=DIRECTORY+f'//MNIST Dataset JPG format//MNIST - JPG - testing//{self.number}//{self.random_file}'
    
    def get_guess(self):
        self.guesses+=1
        image = PillowImage.open(self.imgpath)
        image_tensor = ToTensor()(image).unsqueeze(0)
        result = torch.argmax(self.model(image_tensor).item())
        return result
    
    def test_model(self):
        self.testing=True
        if self.number==self.get_guess():
            self.correct_guesses+=1
        self.accuracy=self.correct_guesses/self.guesses
        self.testing=False
        

#App Start

app=QtWidgets.QApplication(sys.argv)
window=PTAppMainWindow()
window.show()
app.exec_()

#tensor=torch.zeros(3,900,1600)
#print(tensor.shape)