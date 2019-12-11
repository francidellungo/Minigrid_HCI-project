import matplotlib.pyplot as plt
from datetime import datetime
import os

# rewards = [ [100, 0.14483175527943534], [200, 0.4976801641421471], [300, 0.6177665010752288],
# [400, 0.6828613349288711], [500, 0.6971354725477639], [600, 0.7031887414227903], [700, 0.7837288543973534],
# [800, 0.8345397399146711], [900, 0.813907557406059]]

# ! not save the first step= 0


def plot_reward(rewards, save, plot_path):
    # x axis values
    x = [c for [c, _] in rewards]
    # corresponding y axis values
    y = [c for [_, c] in rewards]

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('num episodes')
    # naming the y axis
    plt.ylabel('avg reward')

    # giving a title to my graph
    plt.title('AVG REWARDS')

    # save figure
    if save:
        plot_path_save = os.path.join(plot_path, 'rewards' + str(x[0]) + '_' + str(x[-1]) + '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.png')
        plt.savefig(plot_path_save)
        # plt.savefig('rewards' + str(x[0]) + '_' + str(x[-1]) + '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.png')

    # function to show the plot
    plt.show()
