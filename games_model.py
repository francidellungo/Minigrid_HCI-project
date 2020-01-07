from PyQt5.QtCore import QObject, pyqtSignal
import os

env = 'MiniGrid-Empty-6x6-v0'

games_path = 'games'


class GamesModel(QObject):

    new_game_s = pyqtSignal(str, str, str)  # TODO fix
    game_removed = pyqtSignal(str)
    game_moved = pyqtSignal(str, int)

    moved_up = pyqtSignal(int)
    moved_down = pyqtSignal(int)

    def __init__(self, env):
        super().__init__()
        # self.games = {}
        self.games_list = []
        self.ranked_games = []
        self.games_info = {}  # TODO to complete folder name, name ecc
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
            # self.games.append(('game' + str(idx), game, 'games'))
            self.games_list.append(game)
            # self.games[str(game)] = {'name': 'game' + str(idx), 'list': 'games'}
            self.n_games += 1

    def new_game(self, env, name):
        """
        create and add a new game
        :param name: game name
        :return:
        """
        new_game = {'name': name, 'folder_name': None, 'list': 'games'}
        self.new_game_s.emit(env, name, new_game['folder_name'])
        self.games.append(new_game)
        self.n_games += 1
        # minigrid play game

    def move_game(self, current_list, game_idx):
        """
        move game from a list to  the other (games -> ranking or ranking -> games)
        :param current_list: list where the game is before moving
        :param game_idx: index of the element in the current list
        :return:
        """
        source_list = self.games if current_list == 'games' else self.ranked_games
        dest_list = self.ranked_games if current_list == 'games' else self.games
        game = source_list.pop(game_idx)
        dest_list.append(game)
        self.game_moved.emit(current_list, game_idx)

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
            print(len(self.games_list))
            self.games_list.remove(folder_name)
            # del self.games[str(folder_name)]
        else:
            del self.ranked_games[folder_name]


        # DELETE
        # if current_list == 'games':
        #     print(game_idx, len(self.games))
        #     return self.games.pop(game_idx)
        # else:
        #     print(game_idx, len(self.ranked_games))
        #     return self.ranked_games.pop(game_idx)

    def move_up(self, game_idx):
        # move element up in the rank list
        if game_idx is not 0:
            self.ranked_games[game_idx], self.ranked_games[game_idx-1] = self.ranked_games[game_idx-1], self.ranked_games[game_idx]
            self.moved_up.emit(game_idx)
            return True
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


