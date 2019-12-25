import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QWidget, \
    QGridLayout, QScrollArea, QVBoxLayout

from agent_detail_window import AgentDetailWindow
from agents_model import AgentsModel
from agents_ui import Ui_Agents


class AgentsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Keep a dictionary to map each 'env_tab_widget' into its 'scroll_area_content_layout'
        self.get_content_layout = {}

        # Keep a reference to the AgentDetailWindow
        self.agent_details_window = None

        # Instantiate a model.
        self._model = AgentsModel()

        # Set up the user interface from Designer.
        self.ui = Ui_Agents()
        self.ui.setupUi(self)

        # Connect model signals to slots
        self._model.environment_added.connect(self.add_environment_to_gui)
        self._model.environment_deleted.connect(self.delete_environment_from_gui)
        self._model.agent_added.connect(self.add_agent_to_gui)
        self._model.agent_deleted.connect(self.delete_agent_from_gui)

        self.ui.environments_tabs.removeTab(0) # TODO remove

        # Connect the buttons events to slots
        self.ui.btn_create.clicked.connect(lambda : self._model.add_agent("MiniGrid-Empty-6x6-v0", "Agent 1")) # TODO change
        # self.ui.btn_add_env.clicked.connect()
        # self.ui.btn_delete_env.clicked.connect()

        # Update GUI loading model data
        self.update_gui_from_model()

    def update_gui_from_model(self):
        for env in self._model.agents:
            self.add_environment_to_gui(env)

            for agent in self._model.agents[env]:
                self.add_agent_to_gui(env, agent) # TODO cambiare con riga sotto
                #self.add_agent_to_gui(env, self._model.agents[env][agent]["name"])

    def add_environment_to_gui(self, environment):

        env_tab_widget = QWidget()
        env_tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        env_tab_layout = QGridLayout(env_tab_widget)

        scroll_area = QScrollArea(env_tab_widget)
        scroll_area_content_widget = QWidget()
        scroll_area_content_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        scroll_area.setWidget(scroll_area_content_widget)
        scroll_area_content_layout = QVBoxLayout(scroll_area_content_widget)

        scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        scroll_area.setWidgetResizable(True)

        env_tab_layout.addWidget(scroll_area)

        self.get_content_layout[env_tab_widget] = scroll_area_content_layout
        self.ui.environments_tabs.addTab(env_tab_widget, environment)

    def delete_environment_from_gui(self, environment):
        # TODO implementare
        pass

    def add_agent_to_gui(self, environment, agent):
        hLayout = QHBoxLayout()
        label = QLabel(agent)
        label.setAlignment(Qt.AlignRight)
        btn = QPushButton("info")
        btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn.clicked.connect(lambda _, environment=environment, agent=agent : self.info_click_slot(environment, agent))
        hLayout.addWidget(label)
        hLayout.addWidget(btn)

        self.get_content_layout[self.ui.environments_tabs.currentWidget()].addLayout(hLayout)

    def delete_agent_from_gui(self, agent_key):
        # TODO implementare
        pass

    def add_environment_slot(self):
        # TODO implementare
        pass

    def info_click_slot(self, environment, agent):
        #self.setEnabled(False)
        if self.agent_details_window is not None:
            pos, size = self.agent_details_window.pos(), self.agent_details_window.size()
            self.agent_details_window = AgentDetailWindow(self._model.agents, environment, agent)
            self.agent_details_window.move(pos)
            self.agent_details_window.resize(size)
        else:
            self.agent_details_window = AgentDetailWindow(self._model.agents, environment, agent)
        self.agent_details_window.show()

    def delete_env_click_slot(self):
        # TODO implementare
        pass


if __name__== "__main__":
    app = QApplication(sys.argv)
    window = AgentsWindow()
    window.show()
    sys.exit(app.exec_())
