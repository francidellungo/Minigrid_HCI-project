# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'agent_detail.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Agent(object):
    def setupUi(self, Agent):
        Agent.setObjectName("Agent")
        Agent.resize(884, 450)
        self.centralwidget = QtWidgets.QWidget(Agent)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelStatus = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelStatus.sizePolicy().hasHeightForWidth())
        self.labelStatus.setSizePolicy(sizePolicy)
        self.labelStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStatus.setObjectName("labelStatus")
        self.verticalLayout.addWidget(self.labelStatus)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.labelLoading = QtWidgets.QLabel(self.centralwidget)
        self.labelLoading.setObjectName("labelLoading")
        self.gridLayout.addWidget(self.labelLoading, 0, 1, 1, 1)
        self.progressBarTraining = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBarTraining.sizePolicy().hasHeightForWidth())
        self.progressBarTraining.setSizePolicy(sizePolicy)
        self.progressBarTraining.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.progressBarTraining.setProperty("value", 24)
        self.progressBarTraining.setObjectName("progressBarTraining")
        self.gridLayout.addWidget(self.progressBarTraining, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.SliderTraining = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SliderTraining.sizePolicy().hasHeightForWidth())
        self.SliderTraining.setSizePolicy(sizePolicy)
        self.SliderTraining.setMouseTracking(False)
        self.SliderTraining.setStyleSheet("")
        self.SliderTraining.setMaximum(10)
        self.SliderTraining.setOrientation(QtCore.Qt.Horizontal)
        self.SliderTraining.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.SliderTraining.setTickInterval(1)
        self.SliderTraining.setObjectName("SliderTraining")
        self.gridLayout.addWidget(self.SliderTraining, 1, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 4, 1, 1)
        self.btnPlayPause = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnPlayPause.sizePolicy().hasHeightForWidth())
        self.btnPlayPause.setSizePolicy(sizePolicy)
        self.btnPlayPause.setText("")
        self.btnPlayPause.setObjectName("btnPlayPause")
        self.gridLayout.addWidget(self.btnPlayPause, 0, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalWidget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget.sizePolicy().hasHeightForWidth())
        self.horizontalWidget.setSizePolicy(sizePolicy)
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout.addWidget(self.horizontalWidget)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.labelGame = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelGame.sizePolicy().hasHeightForWidth())
        self.labelGame.setSizePolicy(sizePolicy)
        self.labelGame.setText("")
        self.labelGame.setObjectName("labelGame")
        self.horizontalLayout_4.addWidget(self.labelGame)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.formWidget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.formWidget.sizePolicy().hasHeightForWidth())
        self.formWidget.setSizePolicy(sizePolicy)
        self.formWidget.setObjectName("formWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formWidget)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.txt_name = QtWidgets.QLabel(self.formWidget)
        self.txt_name.setText("")
        self.txt_name.setObjectName("txt_name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_name)
        self.verticalLayout.addWidget(self.formWidget)
        self.horizontalWidget1 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalWidget1.sizePolicy().hasHeightForWidth())
        self.horizontalWidget1.setSizePolicy(sizePolicy)
        self.horizontalWidget1.setObjectName("horizontalWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnDeleteAgent = QtWidgets.QPushButton(self.horizontalWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnDeleteAgent.sizePolicy().hasHeightForWidth())
        self.btnDeleteAgent.setSizePolicy(sizePolicy)
        self.btnDeleteAgent.setAutoFillBackground(False)
        self.btnDeleteAgent.setObjectName("btnDeleteAgent")
        self.horizontalLayout_2.addWidget(self.btnDeleteAgent)
        self.verticalLayout.addWidget(self.horizontalWidget1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.info_verticalLayout = QtWidgets.QVBoxLayout()
        self.info_verticalLayout.setContentsMargins(-1, 0, -1, -1)
        self.info_verticalLayout.setObjectName("info_verticalLayout")
        self.list_title_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.list_title_label.setFont(font)
        self.list_title_label.setObjectName("list_title_label")
        self.info_verticalLayout.addWidget(self.list_title_label)
        self.info_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.info_scrollArea.setWidgetResizable(True)
        self.info_scrollArea.setObjectName("info_scrollArea")
        self.info_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.info_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 426, 362))
        self.info_scrollAreaWidgetContents.setObjectName("info_scrollAreaWidgetContents")
        self.info_verticalLayout_2 = QtWidgets.QVBoxLayout(self.info_scrollAreaWidgetContents)
        self.info_verticalLayout_2.setObjectName("info_verticalLayout_2")
        self.info_scrollArea.setWidget(self.info_scrollAreaWidgetContents)
        self.info_verticalLayout.addWidget(self.info_scrollArea)
        self.horizontalLayout.addLayout(self.info_verticalLayout)
        Agent.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Agent)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 884, 22))
        self.menubar.setObjectName("menubar")
        Agent.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Agent)
        self.statusbar.setObjectName("statusbar")
        Agent.setStatusBar(self.statusbar)

        self.retranslateUi(Agent)
        QtCore.QMetaObject.connectSlotsByName(Agent)

    def retranslateUi(self, Agent):
        _translate = QtCore.QCoreApplication.translate
        Agent.setWindowTitle(_translate("Agent", "MainWindow"))
        self.labelStatus.setText(_translate("Agent", "Status: training"))
        self.labelLoading.setText(_translate("Agent", "x"))
        self.progressBarTraining.setFormat(_translate("Agent", "%v/%m"))
        self.label.setText(_translate("Agent", "Name:"))
        self.btnDeleteAgent.setText(_translate("Agent", "Delete agent"))
        self.list_title_label.setText(_translate("Agent", "Games used for training"))

