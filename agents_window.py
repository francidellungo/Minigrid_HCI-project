import os
import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie

from PyQt5.QtWidgets import *

from agent_detail_window import AgentDetailWindow
from agents_model import AgentsModel
from agents_ui import Ui_Agents
from games_window import GamesController
from trainer import TrainingManager
from utils import get_all_environments


class AgentsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.agents_number = {}

        self.gif = None
        self.sep = "^"

        # Keep a reference to the AgentDetailWindow
        self.agent_details_window = None

        # Instantiate a model.
        self._agents_model = AgentsModel()

        # Set up the user interface from Designer.
        self.ui = Ui_Agents()
        self.ui.setupUi(self)

        # Connect model signals to slots
        self._agents_model.environment_added.connect(self.add_environment_to_gui)
        self._agents_model.environment_deleted.connect(self.delete_environment_from_gui)
        self._agents_model.agent_added.connect(self.add_agent_to_gui)
        self._agents_model.agent_deleted.connect(self.delete_agent_from_gui)

        # Connect the buttons events to slots
        self.ui.btn_create.clicked.connect(self.create_agent_click_slot)
        # self.ui.btn_add_env.clicked.connect()
        # self.ui.btn_delete_env.clicked.connect()

        # create button for add the first environment
        self.row_big_add_env = QWidget()
        self.row_big_add_env_layout = QHBoxLayout(self.row_big_add_env)
        self.row_big_add_env_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        self.big_btn_add_env = QPushButton("Add environment")
        self.big_btn_add_env.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.big_btn_add_env.clicked.connect(self.ask_for_new_environment)
        self.row_big_add_env_layout.addWidget(self.big_btn_add_env)
        self.row_big_add_env_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        self.ui.verticalLayout.addWidget(self.row_big_add_env)

        # Update GUI loading model data
        self.create_gui_from_model()
        self.ui.environments_tabs.setCurrentIndex(0)

        # add environment button
        self.btn_add_env = QPushButton("+")
        self.btn_add_env.clicked.connect(self.ask_for_new_environment)
        self.ui.environments_tabs.setCornerWidget(self.btn_add_env, Qt.TopRightCorner)

    def create_gui_from_model(self):
        agents_envs = self._agents_model.get_environments()
        for env in get_all_environments():
            if env not in agents_envs:
                continue

            self.add_environment_to_gui(env)
            for agent_key in self._agents_model.get_agents(env):
                self.add_agent_to_gui(env, agent_key)

        self.refresh_visibility()

    def refresh_visibility(self):
        envs_exist = len(self._agents_model.get_environments()) > 0
        self.row_big_add_env.setVisible(not envs_exist)
        self.ui.environments_tabs.setVisible(envs_exist)
        self.ui.btn_create.setVisible(envs_exist)

    def ask_for_new_environment(self):
        items = sorted(get_all_environments() - self._agents_model.get_environments())
        env, ok = QInputDialog.getItem(self, "Select an environment to add", "Environment:", items, editable=False)
        if ok:
            self._agents_model.add_environment(env)

    def add_environment_to_gui(self, environment):

        if self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "env_tab_widget") is not None:
            return

        # create tab widget for this environment
        env_tab_widget = QWidget(self.ui.environments_tabs)     # setting the parent is useful to find later the object
        env_tab_widget.setObjectName(environment + self.sep + "env_tab_widget")  # object name is useful to find later the object
        env_tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        env_tab_layout = QGridLayout(env_tab_widget)            # assign a grid layout to this widget

        # create a scroll area and its content widget
        scroll_area = QScrollArea(env_tab_widget)               # scroll area is a child of tab widget
        scroll_area_content_widget = QWidget(scroll_area)       # content of the scroll area is a child of scroll area

        # set properties of scroll area and its content widget
        scroll_area.setWidget(scroll_area_content_widget)
        scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        scroll_area.setWidgetResizable(True)
        scroll_area_content_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        # assign a vertical layout to the content of the scroll area
        scroll_area_content_layout = QVBoxLayout(scroll_area_content_widget)
        scroll_area_content_layout.setObjectName(environment + self.sep + "scroll_area_content_layout") # name is useful to find later the object

        # add label agents
        list_label = QLabel("Agents                     ", env_tab_widget)
        list_label.setAlignment(Qt.AlignRight)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        list_label.setFont(font)
        scroll_area_content_layout.addWidget(list_label)

        # add delete environment button
        row_delete = QWidget(env_tab_widget)
        row_delete.setObjectName(environment + self.sep + "row_delete")
        row_delete_layout = QHBoxLayout(row_delete)
        row_delete_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        btn_delete = QPushButton("Delete this environment", env_tab_widget)
        btn_delete.clicked.connect(lambda _, env=environment: self.delete_environment_click_slot(env))
        btn_delete.setObjectName(environment + self.sep + "btn_delete")
        row_delete_layout.addWidget(btn_delete)
        row_delete_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        scroll_area_content_layout.addWidget(row_delete)

        # add the scroll area to the tab widget, and add the tab widget to the tabs
        env_tab_layout.addWidget(scroll_area)
        self.ui.environments_tabs.addTab(env_tab_widget, environment)

        self.ui.environments_tabs.setCurrentIndex(len(self.ui.environments_tabs)-1)

        self.refresh_visibility()

    def delete_environment_from_gui(self, environment):
        env_tab_widget = self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "env_tab_widget")
        index = self.ui.environments_tabs.indexOf(env_tab_widget)
        self.ui.environments_tabs.removeTab(index)

        self.refresh_visibility()

    def add_agent_to_gui(self, environment, agent):
        env_tab_widget = self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "env_tab_widget")
        scroll_area_content_layout = env_tab_widget.findChild(QLayout, environment + self.sep + "scroll_area_content_layout")

        # create widgets for the new agent
        label_name = QLabel(agent)
        #label_name.setAlignment(Qt.AlignRight)
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        label_name.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        label_loading = QLabel(env_tab_widget)
        label_loading.setAlignment(Qt.AlignRight)
        label_loading.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        label_loading.setObjectName(environment + self.sep + "" + agent + self.sep + "label_loading")
        btn_info = QPushButton("info")
        btn_info.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn_info.clicked.connect(lambda _, env=environment, ag=agent : self.info_click_slot(env, ag))

        # add widgets to row
        row = QWidget(env_tab_widget)
        horiz_layout = QHBoxLayout(row)
        horiz_layout.addItem(spacer)
        horiz_layout.addWidget(label_loading)
        horiz_layout.addWidget(label_name)
        horiz_layout.addWidget(btn_info)
        row.setObjectName(environment + self.sep + agent + self.sep + "row")

        # disable the delete button
        env_tab_widget.findChild(QWidget, environment + self.sep + "row_delete").setVisible(False)

        # add row to the scroll area
        scroll_area_content_layout.addWidget(row)

        # increment count of agents in this environment
        self.agents_number[environment] = self.agents_number.get(environment, 0) + 1

        # update the label loading of this agent
        self.update_agent_on_gui(environment, agent)

        # link slot to agent_updated signal
        self._agents_model.agent_updated.connect(self.update_agent_on_gui)

    def update_agent_on_gui(self, environment, agent_key):
        agent = self._agents_model.get_agent(environment, agent_key)

        if agent is None:
            return

        label_loading = self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "env_tab_widget").\
            findChild(QLabel, environment + self.sep + agent_key + self.sep + "label_loading")

        if not agent.running:
            label_loading.setMovie(None)
            label_loading.setVisible(False)
        else:
            if self.gif is None:
                self.gif = QMovie(os.path.join("img", "loading.gif"))
                self.gif.start()
            label_loading.setMovie(self.gif)
            label_loading.setVisible(True)

    def delete_agent_from_gui(self, environment, agent_key):
        env_tab_widget = self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "env_tab_widget")
        scroll_area_content_layout = env_tab_widget.findChild(QLayout, environment + self.sep + "scroll_area_content_layout")
        row = env_tab_widget.findChild(QWidget, environment + self.sep + agent_key + self.sep + "row")

        # TODO: to fix deletion
        for i in reversed(range(row.layout().count())):
            w = row.layout().itemAt(i).widget()
            if w is not None:
                w.setParent(None)
        scroll_area_content_layout.removeWidget(row)

        self.agents_number[environment] -= 1
        if self.agents_number[environment] == 0:
            self.ui.environments_tabs.findChild(QWidget, environment + self.sep + "row_delete").setVisible(True)

    def info_click_slot(self, environment, agent):
        #self.setEnabled(False)
        if self.agent_details_window is not None:
            pos, size = self.agent_details_window.pos(), self.agent_details_window.size()
            self.agent_details_window = AgentDetailWindow(self._agents_model, environment, agent)
            self.agent_details_window.move(pos)
            self.agent_details_window.resize(size)
        else:
            self.agent_details_window = AgentDetailWindow(self._agents_model, environment, agent)
        self.agent_details_window.show()

    def create_agent_click_slot(self):
        environment = self.ui.environments_tabs.currentWidget().objectName().split(self.sep)[0]
        GamesController(environment, self, self._agents_model)
        self.ui.btn_create.setEnabled(False)

    def delete_environment_click_slot(self, environment):
        button_reply = QMessageBox.question(self, 'Confirm delete', "Are you sure you want to delete the environment {}?".format(environment),
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if button_reply == QMessageBox.Yes:
            self._agents_model.delete_environment(environment)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)

        button_reply = QMessageBox.question(self, 'Confirm exit', "Are you sure you want to close the application?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if button_reply == QMessageBox.Yes:
            TrainingManager.interrupt_all_trainings()
        else:
            a0.ignore()


if __name__== "__main__":
    app = QApplication(sys.argv)
    window = AgentsWindow()
    window.show()
    sys.exit(app.exec_())
