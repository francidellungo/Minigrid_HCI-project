import json
import shutil

from PyQt5.QtCore import QObject, pyqtSignal
import os

env = 'MiniGrid-Empty-6x6-v0'

games_path = 'games'


class GamesModel(QObject):

    new_game_s = pyqtSignal(str, str)
    game_removed = pyqtSignal(str)
    game_moved = pyqtSignal(str, str)

    moved_up = pyqtSignal(str)
    moved_down = pyqtSignal(str)

    def __init__(self, env, agents_model):
        super().__init__()
        # self.games = {}
        self.games_list = []
        self.ranked_games = []
        # self.all_games = {}  # TODO to complete folder name, name ecc
        self.n_games = 0
        self.env = env
        self.load_existing_games(env)
        self.agents_model = agents_model

    def load_existing_games(self, env):
        """
        load existing games from games folder, each game is registered as a dict of dicts:
         {'folder_name1':    {'name': name1, 'list': 'games'},
            'folder_name2':  {'name': name2, 'list': 'games'}
         }
        :param env: current environment used
        :return:
        """
        if os.path.exists(os.path.join(games_path, env)):
            for idx, game in enumerate(os.listdir(os.path.join(games_path, env))):

                with open(os.path.join(games_path, env, game, "game.json"), "rt") as file:
                    j = json.load(file)
                if j["to_delete"]:
                    continue

                self.games_list.append(game)
                # self.all_games[str(game)] = {'name': game}

                # self.games[str(game)] = {'name': 'game' + str(idx), 'list': 'games'}
                self.n_games += 1

    def new_game(self, env, game_key):
        """
        create and add a new game
        :param name: game name
        :return:
        """
        self.games_list.append(game_key)
        print('added new game with game_key:', game_key)
        # new_game = {'name': name, 'folder_name': folder_name, 'list': 'games'}
        self.new_game_s.emit(env, game_key)
        self.n_games += 1

    def move_game(self, current_list, game_name):
        """
        move game from a list to  the other (games -> ranking or ranking -> games)
        :param current_list: list where the game is before moving
        :param game_idx: index of the element in the current list
        :return:
        """
        # print('current_list, game_name: ', current_list, game_name)
        folder_name = game_name
        source_list = self.games_list if current_list == 'games' else self.ranked_games
        dest_list = self.ranked_games if current_list == 'games' else self.games_list
        # game_idx = source_list.index(game_name)
        source_list.remove(game_name)
        dest_list.insert(0, game_name)
        dest_list_name = 'rank' if current_list == 'games' else 'games'

        self.game_moved.emit(dest_list_name, folder_name)

    def remove_game(self, game_key, list_=None):
        """
        remove a game from a list
        :param game_key: game name of the game to be removed
        :param list_: 'games' or 'rank'
        :return:
        """

        if list_ is None:
            list_ = 'games' if game_key in self.games_list else 'ranked'

        if list_ == 'games':
            # print(len(self.games_list))
            self.games_list.remove(game_key)
            # del self.games[str(game_key)]
        else:
            self.ranked_games.remove(game_key)

        removing_dir = os.path.join(games_path, self.env, game_key)
        if self.agents_model.game_used_by_some_agent(self.env, game_key):
            # write "to_delete": true in json
            with open(os.path.join(games_path, self.env, game_key, "game.json"), "rt") as file:
                j = json.load(file)
            j["to_delete"] = True
            with open(os.path.join(games_path, self.env, game_key, "game.json"), "wt") as file:
                json.dump(j, file)
        else:
            # delete game directory
            shutil.rmtree(removing_dir, ignore_errors=True)

        self.game_removed.emit(game_key)

    def move_up(self, game_name):
        # move element up in the rank list
        idx = self.ranked_games.index(game_name)
        assert idx > 0
        self.ranked_games[idx], self.ranked_games[idx - 1] = self.ranked_games[idx - 1], self.ranked_games[idx]
        self.moved_up.emit(game_name)

    def move_down(self, game_name):
        # move element down in the rank list
        idx = self.ranked_games.index(game_name)
        assert idx != len(self.ranked_games) - 1
        self.ranked_games[idx], self.ranked_games[idx + 1] = self.ranked_games[idx + 1], self.ranked_games[idx]
        self.moved_down.emit(game_name)




