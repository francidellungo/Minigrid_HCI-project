import matplotlib.pyplot as plt
import argparse
from gym_minigrid.wrappers import *
from utils import *


def heatmap(data, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    # ax.set_xticklabels(col_labels)
    # ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def calculate_rewards(args_, env, agent_dir=None):
    agent_directions = {
        'right': 0,
        'down': 1,
        'left': 2,
        'up': 3
    }

    dir_to_symbol = {
        'right': '>',
        'down': 'v',
        'left': '<',
        'up': '^'
    }

    env_width = env.grid.width
    env_height = env.grid.height
    reward_net = load_net(args_.reward_net, True)
    net_rewards = []
    directions = []
    for x in range(1, env_width - 1):
        net_rewards_row = []
        net_rewards_row_dir = []
        for y in range(1, env_height - 1):
            # print('position: x: ' + str(x) + ', y: ' + str(y))
            env.agent_pos = (y, x)

            if agent_direction is None:
                # check direction with max prob
                max_prob = [float('-inf'), None]
                for dir in agent_directions.values():
                    env.agent_dir = dir
                    action = env.actions.pickup
                    obs, reward, done, info = env.step(action)
                    if reward_net is not None:
                        net_reward = reward_net(state_filter(obs), torch.tensor([env.step_count])).item()
                        if net_reward > max_prob[0]:
                            max_prob[0] = net_reward
                            max_prob[1] = dir_to_symbol[list(agent_directions.keys())[list(agent_directions.values()).index(dir)]]

                    else:
                        if reward > max_prob[0]:
                            max_prob[0] = reward
                            # max_prob[1] = agent_directions.get(dir)
                            max_prob[1] = dir_to_symbol[list(agent_directions.keys())[list(agent_directions.values()).index(dir)]]

                            # TODO fix
                    # print('max prob : ', str(x), ' ', str(y), max_prob[1])
                net_rewards_row.append(round(max_prob[0], 2))
                net_rewards_row_dir.append(max_prob[1])

            else:
                env.agent_dir = agent_directions[agent_dir]
                action = env.actions.pickup
                obs, reward, done, info = env.step(action)
                if reward_net is not None:
                    net_reward = reward_net(state_filter(obs), torch.tensor([env.step_count])).item()
                    net_rewards_row.append(round(net_reward, 2))
            # env.render()
        net_rewards.append(net_rewards_row)
        directions.append(net_rewards_row_dir)
    # print('net_rewards: ', directions)
    net_rewards = np.array(net_rewards)
    return net_rewards, np.array(directions)


def draw_heatmap(reward, directions):
    fig, ax = plt.subplots()
    # harvest = np.array([[2.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 6.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 0.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 2.3]])
    #
    # rewards = np.array([[1.47, 0.64, 1.18, 1.41],
    #                     [1.2, 0.03, 0.1, 0.64],
    #                     [0.6, 0.2, 0.1, 1.18],
    #                     [1.42, 1.16, 0.58, 0.28]])
    #
    # rewards_down = np.array([[1.66, 0.22, -1.02, -2.46],
    #                     [1.08, -0.18, -1.22, -2.41],
    #                     [0.34, -0.6, -1.37, -2.15],
    #                     [-0.25, -0.87, -1.31, 0.25]])
    #
    # rewards_up = np.array([[-2.26, -2.23, -2.09, -1.67],
    #                     [-0.96, -1.06, -1.20, -1.12],
    #                     [0.30, -0.3, -0.46, -0.59],
    #                     [1.57, 0.93, 0.34, -0.20]])
    #
    # rew = np.array([[-1.197307825088501, -1.1243306398391724, -0.4638795256614685, -0.5922848582267761, 0.9193240404129028],
    #                 [-1.0645592212677002, -2.2605128288269043, -1.197307825088501, -0.026105016469955444, 0.9193240404129028],
    #                 [-1.0645592212677002, -1.0645592212677002, 0.9193240404129028, -0.026105016469955444, 0.9193240404129028],
    #                 [-0.5922848582267761, 0.3383917212486267, -0.4638795256614685, 0.3383917212486267, 0.3383917212486267],
    #                 [-0.5922848582267761, -0.5922848582267761, -1.1243306398391724, -0.5922848582267761, 1.5]])
    #
    # rewd = np.array([[-1.197307825088501, -2.0884687900543213, -0.4638795256614685, -1.674673080444336, -1.1243306398391724],
    #                 [-0.5922848582267761, -0.4638795256614685, -0.026105016469955444, 0.3383917212486267, -1.1243306398391724],
    #                 [-0.9579547643661499, -0.9579547643661499, -1.1243306398391724, 0.3383917212486267, -0.5922848582267761],
    #                 [1.5741075277328491, -0.026105016469955444, 0.3383917212486267, -0.4638795256614685, -0.5922848582267761],
    #                 [0.3383917212486267, 1.5741075277328491, 0.3383917212486267, 0.3383917212486267, 1.5]]
    # )

    # reward = np.array([[-0.2, 0.39, 1.07, 1.59], [-0.71, -0.51, -0.1, 0.13], [-1.17, -1.27, -1.15, -1.16], [-1.63, -2.1, -2.29, -2.38]])

    im = heatmap(reward, ax=ax, cmap="YlGn", cbarlabel="reward")

    for i in range(len(reward)):
        for j in range(len(reward[0])):
            # text = ax.text(j, i, str(round(reward[i, j], 2)) + str('\n') + directions[i, j][0], ha="center", va="center", color="b")
            text = ax.text(j, i, directions[i, j][0], ha="center", va="center", color="c", weight='bold')

    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend", help="Backend to use. Default: qt", default='qt', choices=['qt', 'plt'])
    parser.add_argument("-e", "--env", help="Gym environment to load. Default: MiniGrid-Empty-6x6-v0", default='MiniGrid-Empty-6x6-v0', choices=get_all_environments())
    parser.add_argument("-s", "--seed", type=int, help="Random seed to generate the environment with", default=-1)
    parser.add_argument("-g", "--games_dir", help="Directory where to save games. Default: games aren't saved", default=None)
    parser.add_argument("-p", "--policy_net", help="Policy net to use as agent. Default: no policy_net, the game is the user", default=None)
    parser.add_argument("-r", "--reward_net", help="Reward net to evalute. Default: None", default=None)
    parser.add_argument("-mg", "--max_games", help="Maximum number of games to play. Default: no limits", type=int, default=-1)
    parser.add_argument("-wt", "--waiting_time", help="Policy waiting time (seconds) between moves. Default: 0", type=float, default=0)

    args = parser.parse_args()

    env = gym.make(args.env)

    agent_direction = None
    net_rewards, directions = calculate_rewards(args, env, agent_direction)
    draw_heatmap(net_rewards, directions)


    """ 
    f
    """
    # env.agent_pos = (1, 2)
    # env.agent_dir = 1
    # action = env.actions.pickup
    # obs, reward, done, info = env.step(action)
    # if reward_net is not None:
    #     net_reward = reward_net(state_filter(obs), torch.tensor([env.step_count])).item()
    #     print(round(net_reward, 2))
    #     print(env.__str__())
    # env.render()

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(net_rewards, ax=ax, cmap="YlGn", cbarlabel="harvest [t/year]")
    #
    # fig.tight_layout()
    # plt.show()


#     if args.agent_view:
#         env = RGBImgPartialObsWrapper(env)
#         env = ImgObsWrapper(env)
#
#     policy_net = load_net(args.policy_net, True)
#     reward_net = load_net(args.reward_net, True)
#
#     if args.backend == "qt":
#         app = QApplication(sys.argv)
#         window = QMainWindow()
#         central_widget = QWidget()
#         v_layout = QVBoxLayout(central_widget)
#         widget_game = QLabel("")
#         widget_caption = QLabel("")
#         v_layout.addWidget(widget_game)
#         v_layout.addWidget(widget_caption)
#         window.setCentralWidget(central_widget)
#         redraw = lambda img: (widget_game.setPixmap(nparray_to_qpixmap(img)), widget_caption.setText(env.mission))
#         game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, lambda:..., True, policy_net, args.max_games, args.waiting_time, reward_net)
#         window.keyPressEvent = game.qt_key_handler
#         window.show()
#         sys.exit(app.exec_())
#     elif args.backend == "plt":
#         window = Window('gym_minigrid - ' + args.env)
#         redraw = lambda img: (window.show_img(img), window.set_caption(env.mission))
#         game = Game(env, args.seed, args.agent_view, args.games_dir, redraw, lambda:..., True, policy_net, args.max_games, args.waiting_time, reward_net)
#         window.reg_key_handler(game.plt_key_handler)
#         # Blocking event loop
#         window.show(block=True)
#     else:
#         print("unknown backend")





