
import sys
import pickle
import numpy           as np
from random import randint
import multiprocessing as mp
from   collections     import deque
from   shutil          import copy   as cp
from   os              import remove as rm, mkdir, path
from   math            import ceil
from   time            import sleep
from   pprint          import pprint
from   functools       import partial

from game    import GameState
import globals


"""
This module let a human player play against a net
"""

def evaluate_net_process(net, net_opponent, games_per_proc, it):
    """
    Best network plays some games against the new trained one.
    Returns the sum of ties, wins and loses, where ties are 0,
    wins are 1 and loses -1.
    """
    globals.set_gpu_visible(True)
    from multiprocessing import current_process as cp
    from time import sleep
    from MCTS import MCTS
    from NNet import NNet

    net = NNet('best')
    net_opponent = None

    result = 0

    for game_it in range(it, it+games_per_proc):

        # init environment
        game, first_player = GameState.generate_training_game(game_it+1005)
        mcts = MCTS(net, game.copy())
        #mcts_opponent = MCTS(net_opponent, game.copy())

        player = first_player
        # Game loop
        while True:
            # debug
            #print('root state', mcts.root.state)

            #net turn
            action = np.argmax(mcts.get_pi(player))
            game.next_state(action, player)

            print('\n\n\nnet MOVED\n'+str(game))
            sleep(0.7)
            
            if game.is_finished():
                break


            #action = randint(0,6)
            #print('Playing random', action)
            action = int(input(': '))
            game.next_state(action, -player)

            print('\n\n\nnet_opponent MOVED\n'+str(game))
            sleep(0.7)

            if game.is_finished():
                break

            if not mcts.root.has_children():
                mcts.search(mcts.root, -player)
            assert(mcts.root.children[action])
            mcts.root = mcts.root.children[action]

        #new net was the first to move
        result += 0 if not any(game.valid_moves()) else (1 if game.get_winner()==first_player else -1)

        print(cp(), '=== evaluation ===', '{}/{}'.format(game_it-it+1, games_per_proc))
        #print('.', end='')

    return result



if __name__ == '__main__':
    try:
        it = int(sys.argv[-1])
    except:
        it = 0
    
    evaluate_net_process(None, None, 100, it)
