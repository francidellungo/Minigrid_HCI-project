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
        Agents.resize(499, 458)
        self.centralwidget = QtWidgets.QWidget(Agents)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.environments_tabs = QtWidgets.QTabWidget(self.centralwidget)
        self.environments_tabs.setObjectName("environments_tabs")

        self.first_tab = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.first_tab.setSizePolicy(sizePolicy)
        self.first_tab.setObjectName("first_tab")
        self.first_tab_gridLayout = QtWidgets.QGridLayout(self.first_tab)
        self.first_tab_gridLayout.setObjectName("first_tab_gridLayout")

        self.scroll_area = QtWidgets.QScrollArea(self.first_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.scroll_area.setSizePolicy(sizePolicy)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_area_content_widget = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.scroll_area_content_widget.setSizePolicy(sizePolicy)
        self.scroll_area_content_widget.setObjectName("scrollAreaWidgetContents")
        self.scroll_area_content_layout = QtWidgets.QVBoxLayout(self.scroll_area_content_widget)
        self.scroll_area_content_layout.setObjectName("scrollAreaWidgetContents_layout")

        # self.kwidget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        # self.kwidget.setSizePolicy(sizePolicy)
        # self.kwidget.setObjectName("kwidget")
        # self.kwidget_layout = QtWidgets.QHBoxLayout(self.kwidget)
        # self.kwidget_layout.setObjectName("kwidget_layout")
        # self.klabel = QtWidgets.QLabel(self.kwidget)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        # self.klabel.setSizePolicy(sizePolicy)
        # self.klabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        # self.klabel.setObjectName("klabel")
        # self.kwidget_layout.addWidget(self.klabel)
        # self.kbutton = QtWidgets.QPushButton(self.kwidget)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        # self.kbutton.setSizePolicy(sizePolicy)
        # self.kbutton.setObjectName("kbutton")
        # self.kwidget_layout.addWidget(self.kbutton)
        # self.scrollAreaWidgetContents_layout.addWidget(self.kwidget)
        self.scroll_area.setWidget(self.scroll_area_content_widget)
        self.first_tab_gridLayout.addWidget(self.scroll_area, 0, 0, 1, 1)
        self.environments_tabs.addTab(self.first_tab, "")
        # self.tab_2 = QtWidgets.QWidget()
        # self.tab_2.setObjectName("tab_2")
        # self.environments_tabs.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.environments_tabs)
        self.btn_create = QtWidgets.QPushButton(self.centralwidget)
        self.btn_create.setObjectName("btn_create")
        self.verticalLayout.addWidget(self.btn_create)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        Agents.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Agents)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 499, 22))
        self.menubar.setObjectName("menubar")
        Agents.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Agents)
        self.statusbar.setObjectName("statusbar")
        Agents.setStatusBar(self.statusbar)

        self.retranslateUi(Agents)
        self.environments_tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Agents)

    def retranslateUi(self, Agents):
        _translate = QtCore.QCoreApplication.translate
        Agents.setWindowTitle(_translate("Agents", "Manage your agents"))
        # self.klabel.setText(_translate("Agents", "Agent 1"))
        # self.kbutton.setText(_translate("Agents", "Info"))
        self.environments_tabs.setTabText(self.environments_tabs.indexOf(self.first_tab), _translate("Agents", "MiniGrid-Empty-6x6-v0"))
        #self.environments_tabs.setTabText(self.environments_tabs.indexOf(self.tab_2), _translate("Agents", "MiniGrid-DoorKey-16x16-v0"))
        self.btn_create.setText(_translate("Agents", "Create new agent"))
