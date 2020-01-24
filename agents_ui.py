# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'agents.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Agents(object):
    def setupUi(self, Agents):
        Agents.setObjectName("Agents")
        Agents.resize(1347, 793)
        self.centralwidget = QtWidgets.QWidget(Agents)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.environments_tabs = QtWidgets.QTabWidget(self.centralwidget)
        self.environments_tabs.setObjectName("environments_tabs")
        self.verticalLayout.addWidget(self.environments_tabs)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.btn_create = QtWidgets.QPushButton(self.centralwidget)
        self.btn_create.setObjectName("btn_create")
        self.horizontalLayout.addWidget(self.btn_create)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        Agents.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Agents)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1347, 22))
        self.menubar.setObjectName("menubar")
        Agents.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Agents)
        self.statusbar.setObjectName("statusbar")
        Agents.setStatusBar(self.statusbar)

        self.retranslateUi(Agents)
        self.environments_tabs.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Agents)

    def retranslateUi(self, Agents):
        _translate = QtCore.QCoreApplication.translate
        Agents.setWindowTitle(_translate("Agents", "Manage your agents"))
        self.btn_create.setText(_translate("Agents", "Create new agent"))
