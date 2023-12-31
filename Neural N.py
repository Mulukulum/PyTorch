import subprocess
import sys
import logging
import random
import os
from time import sleep
from PyQt5 import QtGui

from PyQt5.QtCore import QObject

RUNNING_SCRIPT_FROM = sys.argv[0]
DIRECTORY=os.path.dirname(RUNNING_SCRIPT_FROM)
LOGGING_LEVEL= logging.INFO
PREFERRED_DEVICE= 'cpu'
CURRENT_BATCH_SIZE=16

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
    from torchvision.transforms import Compose, PILToTensor, ToTensor
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torch.optim import SGD
except:
    print('Please have PyTorch installed on your system')
    input('Press Enter to Continue')
    sys.exit()

from ui_Main import Ui_MainWindow
from ui_Settings import Ui_SettingsWidget
from ui_TrainSave import Ui_TrainSaveWidget
from ui_LoadTest import Ui_LoadTestWidget

#Main Functions
def get_cuda_information():
    device_count=torch.cuda.device_count()
    return dict([(i,torch.cuda.get_device_name()) for i in range(device_count)])

if USE_CUDA_WHEN_AVAILABLE:
    if get_cuda_information():
        PREFERRED_DEVICE='cuda'


def save_model_state_dict(model_state_dict: dict ,filepath: str ) -> None :
    torch.save(model_state_dict,filepath)
    logging.info(f'File Saved to {filepath}')

def get_model_state_dict(filepath: str, current_device: str ) -> dict : 
    return torch.load(filepath, map_location=torch.device(current_device))
    
class ConvolutionalModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.to(PREFERRED_DEVICE)
        self.mdltype='Conv'
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.LeakyReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10),
        )
    
    def __repr__(self):
        return 'Convolutional'

    def forward(self, x): 
        x = x.to(PREFERRED_DEVICE)
        return self.model(x)

class LinearNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.to(PREFERRED_DEVICE)
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
    
    def __repr__(self):
        return 'Linear'

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(PREFERRED_DEVICE)
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

    #get_model_guess=QtCore.pyqtSignal()

    def __init__(self,frame):
        super().__init__(frame)
        self.ui = Ui_LoadTestWidget()
        self.STOPTESTING=False
        self.ui.setupUi(self)
        self.ModelLoaded=False
        self.ui.TestForLongDuration.setDisabled(True)
        self.ui.TestImageButton.setDisabled(True)
        self.ui.PauseTestingButton.setDisabled(True)
        self.pause=False
        self.model_tester=None
        self.ui.LoadModelButton.clicked.connect(lambda : self.load_and_begin_testing())
        self.ui.TestImageButton.clicked.connect(lambda: self.test_loop())
        self.ui.TestForLongDuration.clicked.connect(lambda : self.TestForLong())
        self.ui.PauseTestingButton.clicked.connect(lambda : self.togglePause())

    def StopTesting(self):
        self.pause=True
        self.STOPTESTING=True
    
    def togglePause(self):
        if self.pause:
            self.pause=False
            self.ui.PauseTestingButton.setText("Pause")
            self.ui.PauseTestingButton.setDisabled(True)
            QtCore.QTimer.singleShot(800,lambda: self.ui.PauseTestingButton.setEnabled(True))
        else:
            self.pause=True
            self.ui.PauseTestingButton.setText("Resume")
            self.ui.PauseTestingButton.setDisabled(True)
            QtCore.QTimer.singleShot(800,lambda: self.ui.PauseTestingButton.setEnabled(True))
        

    def delay(self,n=1):
        dieTime= QtCore.QTime.currentTime().addMSecs(int(1000*n))
        while QtCore.QTime.currentTime() < dieTime:
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents,100)
    
    
    def TestForLong(self):
        self.ui.TestForLongDuration.setDisabled(True)
        self.ui.TestImageButton.hide()
        c=500
        while c>0:
            if self.STOPTESTING:
                break
            while self.pause==True:
                if self.STOPTESTING:
                    break
                self.delay(0.4)
            self.ui.TestImageButton.click()
            self.delay(1)
            c-=1
        self.ui.TestForLongDuration.setEnabled(True)
        self.ui.TestImageButton.show()
        self.ui.TestImageButton.setEnabled(True)
        
    
    def load_model(self):
        #popup to select the model
        filepath = (QtWidgets.QFileDialog.getOpenFileName(self,"Open a saved Model",DIRECTORY+"//Models",'All Files (*)'))[0]
        if filepath=='' : #No file was selected
            return 'nofile'
        else:
            #print(f'{filepath=}')
            try:
                state_dict=get_model_state_dict(filepath,PREFERRED_DEVICE)
                #print(state_dict)
            except :
                logging.info('Couldnt unpickle model')
                QtWidgets.QMessageBox.information(self,"Model could not be opened","Please choose a valid file",QtWidgets.QMessageBox.Ok)
                return
            dirs=filepath.split('/')
            if dirs[-2]=='Conv':
                model_type=ConvolutionalModel
            elif dirs[-2]=='Linear':
                model_type=LinearNetwork
            else:
                box=QtWidgets.QMessageBox.question(self,'Choose model type to be Linear?','The type of the model is unclear. Set it to be a linear Model?',QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
                if box==QtWidgets.QMessageBox.Yes:
                    model_type=LinearNetwork
                else:
                    model_type=ConvolutionalModel

        #If the model loaded, and we know the type, make the model and the tester object
        
        self.model=model_type()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_tester=ModelTester_Manual(model=self.model)
        self.ModelLoaded=True
        self.ui.TestForLongDuration.setDisabled(False)
        self.ui.TestImageButton.setDisabled(False)
        self.ui.ModelNameLabel.setText(dirs[-1])
        self.ui.ModelTypeLabel.setText(str(self.model))
        self.ui.PauseTestingButton.setEnabled(True)

        #Basic setup complete. Now, begin testing model
    
    def load_and_begin_testing(self):
        a=self.load_model()
        if a=='nofile':
            return
        #Gets the current file from the tester and displays it.
        self.ui.PictureLabel.setPixmap(QtGui.QPixmap(self.model_tester.imgpath))
        self.ui.PictureLabel.setScaledContents(True)
        self.ui.CorrectAnswerLabel.setText(str(self.model_tester.number))
        self.ui.CurrentGuessValueLabel.setText(str(self.model_tester.test_model()))
        self.ui.TotalCorrectGuessLabel.setText(str(self.model_tester.correct_guesses) + f'({self.model_tester.accuracy*100}%)' )
        self.ui.TotalGuessesLabel.setText(str(self.model_tester.guesses))
    
    def test_loop(self):
        self.ui.PictureLabel.setPixmap(QtGui.QPixmap(self.model_tester.imgpath))
        self.ui.PictureLabel.setScaledContents(True)
        self.ui.CorrectAnswerLabel.setText(str(self.model_tester.number))
        self.ui.CurrentGuessValueLabel.setText(str(self.model_tester.test_model()))
        self.ui.TotalCorrectGuessLabel.setText(str(self.model_tester.correct_guesses) + f'({self.model_tester.accuracy*100}%)' )
        self.ui.TotalGuessesLabel.setText(str(self.model_tester.guesses))
                                         

class TrainSaveWidget(QtWidgets.QWidget):

    Start_Training=QtCore.pyqtSignal()

    def __init__(self,frame):
        super().__init__(frame)
        self.ui = Ui_TrainSaveWidget()
        self.ui.setupUi(self)
        self.model=LinearNetwork()
        self.ui.SaveModelButton.setDisabled(True)
        self.ui.LearningRateSpinBox.setSingleStep(0.01)
        self.ui.BatchSizeSpinBox.setMinimum(1)
        self.ui.BatchSizeSpinBox.setMaximum(256)
        self.ui.LearningRateSpinBox.setMaximum(1.0)
        self.ui.LearningRateSpinBox.setMinimum(0.0)
        self.ui.EpochCountSpinBox.setMaximum(1000000)
        self.ui.EpochUpdateSpinBox.setMaximum(1000)
        self.ui.EpochCountSpinBox.setMinimum(5)
        self.ui.EpochUpdateSpinBox.setMinimum(1)
        self.ui.BatchSizeSpinBox.setValue(32)
        self.ui.LearningRateSpinBox.setValue(0.01)
        self.ui.LearningRateSpinBox.setDecimals(4)
        self.ui.EpochCountSpinBox.setValue(10)
        self.ui.TrainModelButton.clicked.connect(lambda : self.start_training())
        self.ui.SaveModelButton.clicked.connect(lambda: self.save_model())

    def save_model(self):
        FileName=QtWidgets.QFileDialog.getSaveFileName(self,'Choose Location to save Model',DIRECTORY+f"//Models//{self.model.mdltype}//","Pytorch Models (*.pth)")[0]
        save_model_state_dict(self.model.state_dict(),FileName)
        

    def start_training(self):
        batch_size=self.ui.BatchSizeSpinBox.value()
        learning_rate=self.ui.LearningRateSpinBox.value()
        epochs=self.ui.EpochCountSpinBox.value()
        epoch_update_after_this_many_epochs=self.ui.EpochUpdateSpinBox.value()
        IsConv=self.ui.CreateConvolutionalNeuralNetwork.isChecked()
        if IsConv:
            self.model=ConvolutionalModel()
        else:
            self.model=LinearNetwork()
        self.ui.BatchSizeSpinBox.setDisabled(True)
        self.ui.EpochCountSpinBox.setDisabled(True)
        self.ui.EpochUpdateSpinBox.setDisabled(True)
        self.ui.LearningRateSpinBox.setDisabled(True)
        self.ui.CreateConvolutionalNeuralNetwork.setDisabled(True)
        self.ui.StatusUpdateLabel.append(f'STARTING TRAINING FOR MODEL WITH PARAMETERS : \n{batch_size=}\n{learning_rate=}\n{epochs=}\n{epoch_update_after_this_many_epochs=}\n')
        self.ui.SaveModelButton.setDisabled(True)
        self.ui.TrainModelButton.setDisabled(True)
        trainer=ModelTrainer(model=self.model,batch_size=batch_size,learning_rate=learning_rate,status_report_after=epoch_update_after_this_many_epochs,number_of_epochs=epochs,parent=self)
        trainer.Updated.connect(lambda msg: self.ui.StatusUpdateLabel.append('\n'+msg+'\n'))
        QtWidgets.QMessageBox.information(self,"TRAINING STARTING", "Check the Terminal for updates")
        self.Start_Training.emit()
        trainer.epoch_loops(model=trainer.model,loss_fn=trainer.loss_fn,optimizer=trainer.optimizer,number_of_epochs=trainer.number_of_epochs,learning_rate=trainer.learning_rate,status_report_after=trainer.status_report_after,batch_size=trainer.batch_size)
        self.ui.SaveModelButton.setDisabled(False)
        self.ui.TrainModelButton.setDisabled(False)
        self.ui.BatchSizeSpinBox.setDisabled(False)
        self.ui.EpochCountSpinBox.setDisabled(False)
        self.ui.EpochUpdateSpinBox.setDisabled(False)
        self.ui.LearningRateSpinBox.setDisabled(False)
        self.ui.CreateConvolutionalNeuralNetwork.setDisabled(False)
        #Training Loop started


class PTAppMainWindow(QtWidgets.QMainWindow):

   
    def __init__(self):
        super().__init__()
        self.testwidgetobj=None
        self.TRAIN_SAVE_WIDGET=1
        self.LOAD_TEST_WIDGET=2
        self.CurrentFocus=0
        self.SETTINGS_WIDGET=3
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(960,540)
        self.widgetFrame=QtWidgets.QFrame()
        self.ui.SettingsButton.clicked.connect(lambda : self.showSettingsWidget())
        self.ui.LoadTestButton.clicked.connect(lambda: self.showLoadTestWidget())
        self.ui.TrainSaveButton.clicked.connect(lambda: self.showTrainSaveWidget())
        self.showSettingsWidget()
        
    def closeEvent(self, a0:QtGui.QCloseEvent) -> None:
        if self.testwidgetobj is None:
            ...
        else:
            self.testwidgetobj.STOPTESTING=True
        return super().closeEvent(a0)
    

    def showSettingsWidget(self):
        if self.CurrentFocus==self.SETTINGS_WIDGET:
            return
        else:
            self.CurrentFocus=self.SETTINGS_WIDGET
        self.widgetFrame.deleteLater()
        settings_widget=SettingsWidget(self.ui.MainWindowFrame)
        self.ui.VLayoutForFocusWidget.addWidget(settings_widget)
        self.widgetFrame=settings_widget
    
    def showLoadTestWidget(self):
        if self.CurrentFocus==self.LOAD_TEST_WIDGET:
            return
        else:
            self.CurrentFocus=self.LOAD_TEST_WIDGET
        self.widgetFrame.deleteLater()
        loadtest_widget=LoadTestWidget(self.ui.MainWindowFrame)
        self.testwidgetobj=loadtest_widget
        self.ui.VLayoutForFocusWidget.addWidget(loadtest_widget)
        self.widgetFrame=loadtest_widget
    
    def showTrainSaveWidget(self):
        if self.CurrentFocus==self.TRAIN_SAVE_WIDGET:
            return
        else:
            self.CurrentFocus=self.TRAIN_SAVE_WIDGET
        self.widgetFrame.deleteLater()
        trainsave_widget=TrainSaveWidget(self.ui.MainWindowFrame)
        self.ui.VLayoutForFocusWidget.addWidget(trainsave_widget)
        self.widgetFrame=trainsave_widget
    

class ModelTrainer(QtCore.QObject):

    epoch_status_messages = ['',]
    Updated=QtCore.pyqtSignal(str)

    def __init__(self, model: torch.nn.Module,loss_fn=None,
                 optimizer=None, number_of_epochs: int=10, 
                 learning_rate: float=0.005, batch_size: int=16,
                 status_report_after: int = 1,parent=None):
        super().__init__()
        self.parent=parent
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.number_of_epochs=number_of_epochs
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.status_report_after=status_report_after
        #parent.Start_Training.connect(self.epoch_loops(model=self.model,loss_fn=self.loss_fn,optimizer=self.optimizer,number_of_epochs=self.number_of_epochs,learning_rate=self.learning_rate,status_report_after=self.status_report_after,batch_size=self.batch_size))
        
        
    
    def epoch_loops(self,model: torch.nn.Module,loss_fn=None,
                 optimizer=None, number_of_epochs: int=10, 
                 learning_rate: float=0.005, batch_size: int=16,
                 status_report_after: int = 1 ):
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        if optimizer is None:
            self.optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        DATA=datasets.MNIST(root='mnist', train=True, transform=ToTensor(), download=True)
        DATASET=DataLoader(DATA, batch_size=batch_size)
        model.train()
        model.to(PREFERRED_DEVICE)

        for epoch in range(1, number_of_epochs+1) : 
            print(f'Epoch Number : {epoch}')
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

            if epoch%status_report_after == 0 :
                    self.epoch_status_messages.append(f"Epoch:{epoch} loss = {loss.item()}\n")
                    print(f"Epoch:{epoch} loss = {loss.item()}\n")
                    #self.Updated.emit(f"Epoch:{epoch} loss = {loss.item()}\n")
                    self.parent.ui.StatusUpdateLabel.append(f"Epoch:{epoch} loss = {loss.item()}\n")
                    
                    

    def get_status_messages(self):
        return self.epoch_status_messages

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
        
        self.model.to(PREFERRED_DEVICE)
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
        image_tensor = PILToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.float()
        image_tensor.to(PREFERRED_DEVICE)
        with torch.no_grad():
            self.current_guess = torch.argmax(self.model(image_tensor)).item()
            return self.current_guess
    
    def test_model(self):
        self.testing=True
        self.model.eval()
        guess=self.get_guess()
        if int(self.number)==guess:
            self.correct_guesses+=1
        self.accuracy=self.correct_guesses/self.guesses
        self.testing=False
        self.update_file()
        return guess


#App Start

app=QtWidgets.QApplication(sys.argv)
window=PTAppMainWindow()
window.show()
sys.exit(app.exec_())


#tensor=torch.zeros(3,900,1600)
#print(tensor.shape)