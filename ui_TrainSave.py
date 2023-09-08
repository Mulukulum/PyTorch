# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'TrainLoadoXQKKe.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_TrainSaveWidget(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(588, 384)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.Options = QGridLayout()
        self.Options.setObjectName(u"Options")
        self.LearningRateSpinBox = QDoubleSpinBox(Form)
        self.LearningRateSpinBox.setObjectName(u"LearningRateSpinBox")

        self.Options.addWidget(self.LearningRateSpinBox, 2, 1, 1, 1)

        self.BatchSizeLabel = QLabel(Form)
        self.BatchSizeLabel.setObjectName(u"BatchSizeLabel")

        self.Options.addWidget(self.BatchSizeLabel, 1, 0, 1, 1)

        self.LearningRateLabel = QLabel(Form)
        self.LearningRateLabel.setObjectName(u"LearningRateLabel")

        self.Options.addWidget(self.LearningRateLabel, 2, 0, 1, 1)

        self.TrainModelButton = QPushButton(Form)
        self.TrainModelButton.setObjectName(u"TrainModelButton")

        self.Options.addWidget(self.TrainModelButton, 1, 2, 1, 1)

        self.CreateConvolutionalNeuralNetwork = QCheckBox(Form)
        self.CreateConvolutionalNeuralNetwork.setObjectName(u"CreateConvolutionalNeuralNetwork")

        self.Options.addWidget(self.CreateConvolutionalNeuralNetwork, 2, 2, 1, 1)

        self.UpdateStatusLabel = QLabel(Form)
        self.UpdateStatusLabel.setObjectName(u"UpdateStatusLabel")

        self.Options.addWidget(self.UpdateStatusLabel, 3, 0, 1, 1)

        self.SaveModelButton = QPushButton(Form)
        self.SaveModelButton.setObjectName(u"SaveModelButton")

        self.Options.addWidget(self.SaveModelButton, 0, 2, 1, 1)

        self.BatchSizeSpinBox = QSpinBox(Form)
        self.BatchSizeSpinBox.setObjectName(u"BatchSizeSpinBox")

        self.Options.addWidget(self.BatchSizeSpinBox, 1, 1, 1, 1)

        self.EpochUpdateSpinBox = QSpinBox(Form)
        self.EpochUpdateSpinBox.setObjectName(u"EpochUpdateSpinBox")

        self.Options.addWidget(self.EpochUpdateSpinBox, 3, 1, 1, 1)

        self.LossFunctionDropDown = QComboBox(Form)
        self.LossFunctionDropDown.addItem("")
        self.LossFunctionDropDown.addItem("")
        self.LossFunctionDropDown.addItem("")
        self.LossFunctionDropDown.setObjectName(u"LossFunctionDropDown")

        self.Options.addWidget(self.LossFunctionDropDown, 3, 2, 1, 1)

        self.EpochsLabel = QLabel(Form)
        self.EpochsLabel.setObjectName(u"EpochsLabel")

        self.Options.addWidget(self.EpochsLabel, 0, 0, 1, 1)

        self.EpochCountSpinBox = QSpinBox(Form)
        self.EpochCountSpinBox.setObjectName(u"EpochCountSpinBox")

        self.Options.addWidget(self.EpochCountSpinBox, 0, 1, 1, 1)


        self.verticalLayout.addLayout(self.Options)

        self.StatusUpdateLabel = QTextBrowser(Form)
        self.StatusUpdateLabel.setObjectName(u"StatusUpdateLabel")

        self.verticalLayout.addWidget(self.StatusUpdateLabel)

        self.verticalLayout.setStretch(0, 4)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.BatchSizeLabel.setText(QCoreApplication.translate("Form", u"Batch Size", None))
        self.LearningRateLabel.setText(QCoreApplication.translate("Form", u"Learning Rate", None))
        self.TrainModelButton.setText(QCoreApplication.translate("Form", u"Train Model", None))
        self.CreateConvolutionalNeuralNetwork.setText(QCoreApplication.translate("Form", u"Create Convolutional Neural Network", None))
        self.UpdateStatusLabel.setText(QCoreApplication.translate("Form", u"Update status after this many epochs", None))
        self.SaveModelButton.setText(QCoreApplication.translate("Form", u"Save Model", None))
        self.LossFunctionDropDown.setItemText(0, QCoreApplication.translate("Form", u"This Doesn't work yet, check again later", None))
        self.LossFunctionDropDown.setItemText(1, QCoreApplication.translate("Form", u"SGD (Stochastic Gradient Descent)", None))
        self.LossFunctionDropDown.setItemText(2, QCoreApplication.translate("Form", u"CrossEntropyLoss", None))

        self.EpochsLabel.setText(QCoreApplication.translate("Form", u"Number of Epochs:", None))
        self.StatusUpdateLabel.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
    # retranslateUi

