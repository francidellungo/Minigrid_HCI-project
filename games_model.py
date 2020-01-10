from PyQt5.QtCore import QObject, pyqtSignal
import os

env = 'MiniGrid-Empty-6x6-v0'

games_path = 'games'


class GamesModel(QObject):

    new_game_s = pyqtSignal(str, str)
    game_removed = pyqtSignal(str)
    game_moved = pyqtSignal(str, str)

    moved_up = pyqtSignal(int)
    moved_down = pyqtSignal(int)

    def __init__(self, env):
        super().__init__()
        # self.games = {}
        self.games_list = []
        self.ranked_games = []
        # self.all_games = {}  # TODO to complete folder name, name ecc
        self.n_games = 0
        self.load_existing_games(env)

    def load_existing_games(self, env):
        """
        load existing games from games folder, each game is registered as a dict of dicts:
         {'folder_name1':    {'name': name1, 'list': 'games'},
            'folder_name2':  {'name': name2, 'list': 'games'}
         }
        :param env: current environment used
        :return:
        """
        for idx, game in enumerate(os.listdir(os.path.join(games_path, env))):
            self.games_list.append(game)
            # self.all_games[str(game)] = {'name': game}

            # self.games[str(game)] = {'name': 'game' + str(idx), 'list': 'games'}
            self.n_games += 1

    def new_game(self, env, folder_name, game_name):
        """
        create and add a new game
        :param name: game name
        :return:
        """
        self.games_list.append(folder_name)
        print(folder_name, 'folder_name')
        # new_game = {'name': name, 'folder_name': folder_name, 'list': 'games'}
        self.new_game_s.emit(env, folder_name)
        self.n_games += 1

    def move_game(self, current_list, game_name):
        """
        move game from a list to  the other (games -> ranking or ranking -> games)
        :param current_list: list where the game is before moving
        :param game_idx: index of the element in the current list
        :return:
        """
        print('current_list, game_name: ', current_list, game_name)
        folder_name = game_name
        source_list = self.games_list if current_list == 'games' else self.ranked_games
        dest_list = self.ranked_games if current_list == 'games' else self.games_list
        # game_idx = source_list.index(game_name)
        source_list.remove(game_name)
        dest_list.append(game_name)
        dest_list_name = 'rank' if current_list == 'games' else 'games'

        self.game_moved.emit(dest_list_name, folder_name)


        # TODO emit signal list changed

    def remove_game(self, folder_name, list_):
        """
        remove a game from a list
        :param folder_name: game name of the game to be removed
        :param list_: 'games' or 'rank'
        :return:
        """
        # TODO are you sure you want to delete?

        self.game_removed.emit(folder_name)

        if list_ == 'games':
            # print(len(self.games_list))
            self.games_list.remove(folder_name)
            # del self.games[str(folder_name)]
        else:
            self.ranked_games.remove(folder_name)

    def move_up(self, game_name, list_):
        # move element up in the rank list
        # if list_ == 'games':
        #     idx = self.games_list.index(game_name)
        #     assert idx > 0
        #     # swap elements in list
        #     self.games_list[idx],  self.games_list[idx-1] = self.games_list[idx-1], self.games_list[idx]
        # else:
        #     idx = self.ranked_games.index(game_name)
        #     assert idx > 0

        return False

    def move_down(self, game_idx):
        # move element down in the rank list
        if game_idx < len(self.ranked_games) - 1:
            self.ranked_games[game_idx], self.ranked_games[game_idx + 1] = self.ranked_games[game_idx + 1], self.ranked_games[game_idx]
            self.moved_down.emit(game_idx)
            return True
        return False
#
# class Game(QObject):
#
#     def __init__(self, game_name):
#         super().__init__()
#         self.name = game_name
#         self.image = None
#         self.info = None
#         self.my_list = 'games'  # my_list == 'games' or 'rank'

    # @property
    # def name(self):
    #     return self.name
    #
    # @name.setter
    # def name(self, name_):
    #     self.name = name_


# class GamesList(list):
#     def __init__(self):
#         super().__init__()
#
#     def append(self, object: _T) -> None:
#         super().append(object)
#
#     def pop(self, index: int = ...) -> _T:
#         return super().pop(index)
#
#     def remove(self, object: _T) -> None:
#         super().remove(object)
#


