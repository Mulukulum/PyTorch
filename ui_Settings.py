# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SettingsXCBmHV.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_SettingsWidget(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(517, 357)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.CudaBox = QCheckBox(Form)
        self.CudaBox.setObjectName(u"CudaBox")

        self.verticalLayout.addWidget(self.CudaBox)

        self.AboutTheProject = QTextBrowser(Form)
        self.AboutTheProject.setObjectName(u"AboutTheProject")
        self.AboutTheProject.setReadOnly(True)

        self.verticalLayout.addWidget(self.AboutTheProject)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.CudaBox.setText(QCoreApplication.translate("Form", u"Use CUDA if Available (Uses only CPU if this is un-checkd)", None))
        self.AboutTheProject.setHtml(QCoreApplication.translate("Form", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt;\">What is this app?</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:24pt;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">I made this to get into a research project with a prof. Its mostly meant to demo the difference between 2 kinds of "
                        "NNs. Convolutional and your basic linear networks. The idea is that Conv's are better for pattern recognition and the basic ones should have poor accuracy but I haven't tested it so i'm not sure.</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">It gets the MNIST numbers dataset and trains models on it. The goal of making this was to have a way to see the difference between models and how different optimizer/loss/activation functions can change the outputs. </span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p align=\"center\" sty"
                        "le=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">If the selections for the optims/loss/activation functions aren't there in this version, then it might be there on my </span><a href=\"https://github.com/Mulukulum\"><span style=\" font-size:12pt; text-decoration: underline; color:#0000ff;\">github</span></a><span style=\" font-size:12pt;\"> in a repo called PyTorch. Time was pretty constrained because of college stuff but hopefully it'll be there by the time you check it out.</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">You should be able to save and load models as well, and test and s"
                        "ee how accurate each model is by varying parameters like learning rate, epochs etc.</span></p></body></html>", None))
    # retranslateUi

