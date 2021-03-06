# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scrollbar_v2.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1125, 660)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.buttons_Hlayout = QtWidgets.QHBoxLayout()
        self.buttons_Hlayout.setContentsMargins(-1, -1, -1, 0)
        self.buttons_Hlayout.setObjectName("buttons_Hlayout")
        self.new_game_pb = QtWidgets.QPushButton(self.centralwidget)
        self.new_game_pb.setObjectName("new_game_pb")
        self.buttons_Hlayout.addWidget(self.new_game_pb, 0, QtCore.Qt.AlignRight)
        self.train_pb = QtWidgets.QPushButton(self.centralwidget)
        self.train_pb.setObjectName("train_pb")
        self.buttons_Hlayout.addWidget(self.train_pb, 0, QtCore.Qt.AlignRight)
        self.gridLayout_4.addLayout(self.buttons_Hlayout, 8, 0, 1, 1)
        self.main_horizontalLayout = QtWidgets.QHBoxLayout()
        self.main_horizontalLayout.setContentsMargins(-1, 0, 0, 0)
        self.main_horizontalLayout.setObjectName("main_horizontalLayout")
        self.games_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.games_scrollArea.sizePolicy().hasHeightForWidth())
        self.games_scrollArea.setSizePolicy(sizePolicy)
        self.games_scrollArea.setWidgetResizable(True)
        self.games_scrollArea.setObjectName("games_scrollArea")
        self.games_verticalW = QtWidgets.QWidget()
        self.games_verticalW.setGeometry(QtCore.QRect(0, 0, 548, 534))
        self.games_verticalW.setObjectName("games_verticalW")
        self.games_verticalLayout = QtWidgets.QVBoxLayout(self.games_verticalW)
        self.games_verticalLayout.setObjectName("games_verticalLayout")
        self.games_scrollArea.setWidget(self.games_verticalW)
        self.main_horizontalLayout.addWidget(self.games_scrollArea)
        self.game_ranking_scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.game_ranking_scrollArea.sizePolicy().hasHeightForWidth())
        self.game_ranking_scrollArea.setSizePolicy(sizePolicy)
        self.game_ranking_scrollArea.setWidgetResizable(True)
        self.game_ranking_scrollArea.setObjectName("game_ranking_scrollArea")
        self.ranking_verticalW = QtWidgets.QWidget()
        self.ranking_verticalW.setGeometry(QtCore.QRect(0, 0, 547, 534))
        self.ranking_verticalW.setObjectName("ranking_verticalW")
        self.ranking_verticalLayout = QtWidgets.QVBoxLayout(self.ranking_verticalW)
        self.ranking_verticalLayout.setObjectName("ranking_verticalLayout")
        self.game_ranking_scrollArea.setWidget(self.ranking_verticalW)
        self.main_horizontalLayout.addWidget(self.game_ranking_scrollArea)
        self.gridLayout_4.addLayout(self.main_horizontalLayout, 6, 0, 1, 1)
        self.titles_Hlayout = QtWidgets.QHBoxLayout()
        self.titles_Hlayout.setContentsMargins(-1, -1, -1, 0)
        self.titles_Hlayout.setObjectName("titles_Hlayout")
        self.game_list_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.game_list_title.setFont(font)
        self.game_list_title.setObjectName("game_list_title")
        self.titles_Hlayout.addWidget(self.game_list_title)
        self.game_ranking_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.game_ranking_title.setFont(font)
        self.game_ranking_title.setObjectName("game_ranking_title")
        self.titles_Hlayout.addWidget(self.game_ranking_title)
        self.gridLayout_4.addLayout(self.titles_Hlayout, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1125, 22))
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(True)
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
        self.new_game_pb.setText(_translate("MainWindow", "New game"))
        self.train_pb.setText(_translate("MainWindow", "Train"))
        self.game_list_title.setText(_translate("MainWindow", "All games"))
        self.game_ranking_title.setText(_translate("MainWindow", "Ranked games (best to worst)"))
