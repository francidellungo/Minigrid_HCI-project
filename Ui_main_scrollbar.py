# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_scrollbar.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(910, 785)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.games_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.games_scrollArea.sizePolicy().hasHeightForWidth())
        self.games_scrollArea.setSizePolicy(sizePolicy)
        self.games_scrollArea.setSizeIncrement(QtCore.QSize(0, 0))
        self.games_scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.games_scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.games_scrollArea.setWidgetResizable(False)
        self.games_scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.games_scrollArea.setObjectName("games_scrollArea")
        self.scrollAreaWidgetContents_7 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_7.setGeometry(QtCore.QRect(0, 36, 440, 586))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents_7.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents_7.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents_7.setObjectName("scrollAreaWidgetContents_7")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.scrollAreaWidgetContents_7)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 0, 401, 556))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.games_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.games_verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.games_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.games_verticalLayout.setObjectName("games_verticalLayout")
        self.game1_horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.game1_horizontalLayout_2.setObjectName("game1_horizontalLayout_2")
        self.game1_name = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.game1_name.setObjectName("game1_name")
        self.game1_horizontalLayout_2.addWidget(self.game1_name)
        self.image_1 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.image_1.setObjectName("image_1")
        self.game1_horizontalLayout_2.addWidget(self.image_1)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_6.setObjectName("pushButton_6")
        self.game1_horizontalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.game1_horizontalLayout_2.addWidget(self.pushButton_7)
        self.games_verticalLayout.addLayout(self.game1_horizontalLayout_2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.game2_name = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.game2_name.setObjectName("game2_name")
        self.horizontalLayout_6.addWidget(self.game2_name)
        self.image_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.image_2.setObjectName("image_2")
        self.horizontalLayout_6.addWidget(self.image_2)
        self.pushButton_8 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_6.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_9.setObjectName("pushButton_9")
        self.horizontalLayout_6.addWidget(self.pushButton_9)
        self.games_verticalLayout.addLayout(self.horizontalLayout_6)
        self.games_scrollArea.setWidget(self.scrollAreaWidgetContents_7)
        self.horizontalLayout.addWidget(self.games_scrollArea)
        self.ranking_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ranking_scrollArea.sizePolicy().hasHeightForWidth())
        self.ranking_scrollArea.setSizePolicy(sizePolicy)
        self.ranking_scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ranking_scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.ranking_scrollArea.setWidgetResizable(False)
        self.ranking_scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.ranking_scrollArea.setObjectName("ranking_scrollArea")
        self.scrollAreaWidgetContents_6 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_6.setGeometry(QtCore.QRect(0, 36, 440, 586))
        self.scrollAreaWidgetContents_6.setObjectName("scrollAreaWidgetContents_6")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.scrollAreaWidgetContents_6)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 401, 556))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.ranking_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.ranking_verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.ranking_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.ranking_verticalLayout.setObjectName("ranking_verticalLayout")
        self.game1_horizontalLayout = QtWidgets.QHBoxLayout()
        self.game1_horizontalLayout.setObjectName("game1_horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.game1_horizontalLayout.addWidget(self.label)
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.game1_horizontalLayout.addWidget(self.pushButton_5)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.game1_horizontalLayout.addWidget(self.pushButton_4)
        self.ranking_verticalLayout.addLayout(self.game1_horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.ranking_verticalLayout.addLayout(self.horizontalLayout_3)
        self.ranking_scrollArea.setWidget(self.scrollAreaWidgetContents_6)
        self.horizontalLayout.addWidget(self.ranking_scrollArea)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 1, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.new_game_pb = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.new_game_pb.setFont(font)
        self.new_game_pb.setObjectName("new_game_pb")
        self.horizontalLayout_5.addWidget(self.new_game_pb, 0, QtCore.Qt.AlignRight)
        self.train_pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.train_pushButton.setFont(font)
        self.train_pushButton.setCheckable(False)
        self.train_pushButton.setAutoDefault(False)
        self.train_pushButton.setDefault(False)
        self.train_pushButton.setFlat(False)
        self.train_pushButton.setObjectName("train_pushButton")
        self.horizontalLayout_5.addWidget(self.train_pushButton, 0, QtCore.Qt.AlignRight)
        self.gridLayout.addLayout(self.horizontalLayout_5, 3, 1, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.games_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.games_label.setFont(font)
        self.games_label.setObjectName("games_label")
        self.horizontalLayout_7.addWidget(self.games_label)
        self.ranking_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.ranking_label.setFont(font)
        self.ranking_label.setObjectName("ranking_label")
        self.horizontalLayout_7.addWidget(self.ranking_label)
        self.gridLayout.addLayout(self.horizontalLayout_7, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 910, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.game1_name.setText(_translate("MainWindow", "game 1 "))
        self.image_1.setText(_translate("MainWindow", "image 1"))
        self.pushButton_6.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_7.setText(_translate("MainWindow", "PushButton"))
        self.game2_name.setText(_translate("MainWindow", "game 2"))
        self.image_2.setText(_translate("MainWindow", "image2"))
        self.pushButton_8.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_9.setText(_translate("MainWindow", "PushButton"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_5.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_4.setText(_translate("MainWindow", "PushButton"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_3.setText(_translate("MainWindow", "PushButton"))
        self.new_game_pb.setText(_translate("MainWindow", "New Game"))
        self.train_pushButton.setText(_translate("MainWindow", "Train"))
        self.games_label.setText(_translate("MainWindow", "Games list"))
        self.ranking_label.setText(_translate("MainWindow", "Ranking games"))

