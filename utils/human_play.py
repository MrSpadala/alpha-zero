
import sys
import numpy as np
from random import randint, seed
from time   import sleep, time

import globals

seed(time())

"""
This module let a human player play against a net
"""

def human_play(net_checkpoint):
    globals.set_gpu_visible(True)
    from time import sleep
    from MCTS import MCTS
    from NNet import NNet
    from game import GameState

    net = NNet(net_checkpoint)

    # init environment
    game, first_player = GameState.generate_training_game(randint(0,10000))
    mcts = MCTS(net, game.copy())

    player = first_player
    # Game loop
    while True:

        #net turn
        action = np.argmax(mcts.get_pi(player))
        game.next_state(action, player)

        print('\n\n\nnet MOVED\n'+str(game))
        sleep(0.7)

        if game.is_finished():
            break

        action = int(input(': '))
        game.next_state(action, -player)

        print('\n\n\nyou MOVED\n'+str(game))
        sleep(0.7)

        if game.is_finished():
            break

        if not mcts.root.has_children():
            mcts.search(mcts.root, -player)
        assert(mcts.root.children[action])
        mcts.root = mcts.root.children[action]

    print('Game over')



def main(net):
    net = net[:-3] if net.endswith('.h5') else net

    while True:
        human_play(net)


