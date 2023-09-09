# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LoadTestbFjhVe.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_LoadTestWidget(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(504, 359)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.OptionsGridLayout = QGridLayout()
        self.OptionsGridLayout.setObjectName(u"OptionsGridLayout")
        self.LoadModelButton = QPushButton(Form)
        self.LoadModelButton.setObjectName(u"LoadModelButton")

        self.OptionsGridLayout.addWidget(self.LoadModelButton, 0, 0, 1, 2)

        self.ModelNameLabel = QLabel(Form)
        self.ModelNameLabel.setObjectName(u"ModelNameLabel")

        self.OptionsGridLayout.addWidget(self.ModelNameLabel, 1, 1, 1, 1)

        self.PauseTestingButton = QPushButton(Form)
        self.PauseTestingButton.setObjectName(u"PauseTestingButton")

        self.OptionsGridLayout.addWidget(self.PauseTestingButton, 0, 2, 1, 1)

        self.ModelTypeLabel = QLabel(Form)
        self.ModelTypeLabel.setObjectName(u"ModelTypeLabel")

        self.OptionsGridLayout.addWidget(self.ModelTypeLabel, 2, 1, 1, 1)

        self.ModelType = QLabel(Form)
        self.ModelType.setObjectName(u"ModelType")

        self.OptionsGridLayout.addWidget(self.ModelType, 2, 0, 1, 1)

        self.ModelName = QLabel(Form)
        self.ModelName.setObjectName(u"ModelName")

        self.OptionsGridLayout.addWidget(self.ModelName, 1, 0, 1, 1)

        self.TestForLongDuration = QPushButton(Form)
        self.TestForLongDuration.setObjectName(u"TestForLongDuration")

        self.OptionsGridLayout.addWidget(self.TestForLongDuration, 1, 2, 1, 1)

        self.TestImageButton = QPushButton(Form)
        self.TestImageButton.setObjectName(u"TestImageButton")

        self.OptionsGridLayout.addWidget(self.TestImageButton, 2, 2, 1, 1)


        self.verticalLayout.addLayout(self.OptionsGridLayout)

        self.PictureLabel = QLabel(Form)
        self.PictureLabel.setObjectName(u"PictureLabel")

        self.verticalLayout.addWidget(self.PictureLabel,alignment=Qt.AlignCenter)

        self.StatisticsGridLayout = QGridLayout()
        self.StatisticsGridLayout.setObjectName(u"StatisticsGridLayout")
        self.TotalGuessesLabel = QLabel(Form)
        self.TotalGuessesLabel.setObjectName(u"TotalGuessesLabel")

        self.StatisticsGridLayout.addWidget(self.TotalGuessesLabel, 0, 1, 1, 1)

        self.ActualNumber = QLabel(Form)
        self.ActualNumber.setObjectName(u"ActualNumber")

        self.StatisticsGridLayout.addWidget(self.ActualNumber, 1, 2, 1, 1)

        self.NoOfCorrectGuesses = QLabel(Form)
        self.NoOfCorrectGuesses.setObjectName(u"NoOfCorrectGuesses")

        self.StatisticsGridLayout.addWidget(self.NoOfCorrectGuesses, 1, 0, 1, 1)

        self.TotalCorrectGuessLabel = QLabel(Form)
        self.TotalCorrectGuessLabel.setObjectName(u"TotalCorrectGuessLabel")

        self.StatisticsGridLayout.addWidget(self.TotalCorrectGuessLabel, 1, 1, 1, 1)

        self.TotalNoofGuesses = QLabel(Form)
        self.TotalNoofGuesses.setObjectName(u"TotalNoofGuesses")

        self.StatisticsGridLayout.addWidget(self.TotalNoofGuesses, 0, 0, 1, 1)

        self.CurrentGuess = QLabel(Form)
        self.CurrentGuess.setObjectName(u"CurrentGuess")

        self.StatisticsGridLayout.addWidget(self.CurrentGuess, 0, 2, 1, 1)

        self.CurrentGuessValueLabel = QLabel(Form)
        self.CurrentGuessValueLabel.setObjectName(u"CurrentGuessValueLabel")

        self.StatisticsGridLayout.addWidget(self.CurrentGuessValueLabel, 0, 3, 1, 1)

        self.CorrectAnswerLabel = QLabel(Form)
        self.CorrectAnswerLabel.setObjectName(u"CorrectAnswerLabel")

        self.StatisticsGridLayout.addWidget(self.CorrectAnswerLabel, 1, 3, 1, 1)


        self.verticalLayout.addLayout(self.StatisticsGridLayout)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.LoadModelButton.setText(QCoreApplication.translate("Form", u"Load Model", None))
        self.ModelNameLabel.setText("")
        self.PauseTestingButton.setText(QCoreApplication.translate("Form", u"Pause Testing", None))
        self.ModelTypeLabel.setText("")
        self.ModelType.setText(QCoreApplication.translate("Form", u"Model Type :", None))
        self.ModelName.setText(QCoreApplication.translate("Form", u"Model Name : ", None))
        self.TestForLongDuration.setText(QCoreApplication.translate("Form", u"Test for A Long Duration", None))
        self.TestImageButton.setText(QCoreApplication.translate("Form", u"Test another Image", None))
        self.PictureLabel.setText("")
        self.TotalGuessesLabel.setText("")
        self.ActualNumber.setText(QCoreApplication.translate("Form", u"Actual Number : ", None))
        self.NoOfCorrectGuesses.setText(QCoreApplication.translate("Form", u"Number of Correct Guesses : ", None))
        self.TotalCorrectGuessLabel.setText("")
        self.TotalNoofGuesses.setText(QCoreApplication.translate("Form", u"Total number of guesses : ", None))
        self.CurrentGuess.setText(QCoreApplication.translate("Form", u"Current Guess : ", None))
        self.CurrentGuessValueLabel.setText("")
        self.CorrectAnswerLabel.setText("")
    # retranslateUi