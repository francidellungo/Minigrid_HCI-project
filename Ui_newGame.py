# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'newGame.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_new_game_Dialog(object):
    def setupUi(self, new_game_Dialog):
        new_game_Dialog.setObjectName("new_game_Dialog")
        new_game_Dialog.setWindowModality(QtCore.Qt.NonModal)
        new_game_Dialog.setEnabled(True)
        new_game_Dialog.resize(495, 352)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(new_game_Dialog.sizePolicy().hasHeightForWidth())
        new_game_Dialog.setSizePolicy(sizePolicy)
        new_game_Dialog.setTabletTracking(False)
        new_game_Dialog.setAcceptDrops(False)
        new_game_Dialog.setAutoFillBackground(False)
        new_game_Dialog.setSizeGripEnabled(False)
        new_game_Dialog.setModal(False)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(new_game_Dialog)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.game_label = QtWidgets.QLabel(new_game_Dialog)
        self.game_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.game_label.setObjectName("game_label")
        self.verticalLayout_2.addWidget(self.game_label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.game_buttonBox = QtWidgets.QDialogButtonBox(new_game_Dialog)
        self.game_buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.game_buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.game_buttonBox.setCenterButtons(False)
        self.game_buttonBox.setObjectName("game_buttonBox")
        self.verticalLayout_2.addWidget(self.game_buttonBox)

        self.retranslateUi(new_game_Dialog)
        self.game_buttonBox.accepted.connect(new_game_Dialog.accept)
        self.game_buttonBox.rejected.connect(new_game_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(new_game_Dialog)

    def retranslateUi(self, new_game_Dialog):
        _translate = QtCore.QCoreApplication.translate
        new_game_Dialog.setWindowTitle(_translate("new_game_Dialog", "New Game"))
        self.game_label.setText(_translate("new_game_Dialog", "image game "))

