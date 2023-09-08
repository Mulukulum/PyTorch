# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainnmFIvD.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(803, 512)
        MainWindow.setMinimumSize(QSize(800, 600))
        self.MainWindowWidget = QWidget(MainWindow)
        self.MainWindowWidget.setObjectName(u"MainWindowWidget")
        self.gridLayout = QGridLayout(self.MainWindowWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.MainScrollArea = QScrollArea(self.MainWindowWidget)
        self.MainScrollArea.setObjectName(u"MainScrollArea")
        self.MainScrollArea.setWidgetResizable(True)
        self.MainScrollAreaContents = QWidget()
        self.MainScrollAreaContents.setObjectName(u"MainScrollAreaContents")
        self.MainScrollAreaContents.setGeometry(QRect(0, 0, 783, 492))
        self.verticalLayout_2 = QVBoxLayout(self.MainScrollAreaContents)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.OptionButtons = QHBoxLayout()
        self.OptionButtons.setObjectName(u"OptionButtons")
        self.TrainSaveButton = QPushButton(self.MainScrollAreaContents)
        self.TrainSaveButton.setObjectName(u"TrainSaveButton")

        self.OptionButtons.addWidget(self.TrainSaveButton)

        self.LoadTestButton = QPushButton(self.MainScrollAreaContents)
        self.LoadTestButton.setObjectName(u"LoadTestButton")

        self.OptionButtons.addWidget(self.LoadTestButton)

        self.SettingsButton = QPushButton(self.MainScrollAreaContents)
        self.SettingsButton.setObjectName(u"SettingsButton")

        self.OptionButtons.addWidget(self.SettingsButton)


        self.verticalLayout_2.addLayout(self.OptionButtons)

        self.MainWindowFrame = QFrame(self.MainScrollAreaContents)
        self.MainWindowFrame.setObjectName(u"MainWindowFrame")
        self.gridLayout_2 = QGridLayout(self.MainWindowFrame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.VLayoutForFocusWidget = QVBoxLayout()
        self.VLayoutForFocusWidget.setObjectName(u"VLayoutForFocusWidget")

        self.gridLayout_2.addLayout(self.VLayoutForFocusWidget, 0, 0, 1, 1)


        self.verticalLayout_2.addWidget(self.MainWindowFrame)

        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 8)
        self.MainScrollArea.setWidget(self.MainScrollAreaContents)

        self.gridLayout.addWidget(self.MainScrollArea, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.MainWindowWidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.TrainSaveButton.setText(QCoreApplication.translate("MainWindow", u"Train and Save Model", None))
        self.LoadTestButton.setText(QCoreApplication.translate("MainWindow", u"Load and Test Model", None))
        self.SettingsButton.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
    # retranslateUi
