
import sys
import pickle
import numpy           as np
import multiprocessing as mp
from   ctypes          import c_int
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
This module contains the whole train pipeline
"""


train_args = globals.dotdict({
    #number of iterations
    'iters' : 1000,

    #parallel processes
    'parallel_num' : 4,

    #number of self plays before training. If it is not divisible
    #by 'parallel_num', it will be rounded up
    'num_self_games' : 1500,

    #games are divided into chink to be consumed by workers 
    'games_per_chunk' : 7,

    #circa the number of evaluation games after training. If it is not divisible
    #by 'parallel_num', it will be rounded up
    'num_eval_games' : 48,

    #if you want to accept the model skipping evaluation phase 
    'always_accept_model' : True,

    #Performs ONLY selfplays using evaluate_net function
    'evaluate_only' : False,

    #Show moves during evaluating. If true it sets parallel_num to 1
    'show_moves' : False,
})

examples = []  #debug


def train(good_luck=False):
    """
    Train procedure:
        0) Try to resume the training if it has been stopped
        1) Self play
        2) Train the net based on the selfplay
        3) Evaluate the trained net
    """
    if not good_luck:
        raise Exception('You will need it')

    #init file and folders for training
    globals.init_training_stuff()

    # Resume training
    iteration = globals.get_last_iteration()

    for it in range(iteration, train_args.iters):
        print('Starting iteration ',it)

        # Self play
        examples = selfplay()
        examples = GameState.all_symmetries(examples)
        globals.dump_selfplay_data(examples)


        # Train
        train_net(examples)

        #raise Exception('STOP HERE')


        # Evaluate
        score = evaluate_net(net=str(globals.get_last_iteration()+1), net_opponent='best') if not train_args.always_accept_model else np.inf
        # If new net scored positively accept new model
        if score >= 0:
            print('Accepting model, score', score)
            globals.accept_model()
        else:
            print('Rejecting model, score', score)


        # Update iteration file to keep track of iterations done
        globals.update_iteration_file(it+1)


# TEMP
#TMP = True
# END TEMP

def selfplay():
    """
    Performs train_args.num_self_games number of self plays distributed over
    train_args.parallel_num parallel processes
    """

    chunk_total   = ceil(train_args.num_self_games / train_args.games_per_chunk)
    chunk_counter = mp.Value(c_int, chunk_total)  #decremental counter
    all_moves_queue = mp.Queue()

    '''
    # TEMP
    global TMP
    if TMP:
        chunk_counter.value = 110
    # END TEMP
    '''

    processes = [None] * train_args.parallel_num

    #p = mp.Process(target=train_net_process, args=(examples,))
    for i in range(train_args.parallel_num):
        processes[i] = mp.Process(target=selfplay_process, args=(chunk_counter, train_args.games_per_chunk, all_moves_queue,))
        processes[i].start()
    
    all_moves = []
    for p in processes:
        #p.join()  NO! It deadlocks
        all_moves += all_moves_queue.get()
    
    all_moves_queue.close()


    '''
    #TEMP
    if TMP:
        from pickle import load
        all_moves += load(open('./tmp/6.pickle', 'rb'))
        all_moves += load(open('./tmp/7.pickle', 'rb'))
        all_moves += load(open('./tmp/8.pickle', 'rb'))
        all_moves += load(open('./tmp/9.pickle', 'rb'))
        TMP = False
    #END TEMP
    '''

    #print(np.asarray(all_moves).shape)   #debug purposes, should be (x, 3)
    return all_moves
    

def selfplay_process(chunk_counter, games_per_chunk, all_moves):
    """
    Process that plays games_per_proc games
    """
    globals.set_gpu_visible(False)
    from multiprocessing import current_process as cp
    from MCTS import MCTS
    from NNet import NNet
    from pickle import dump
    from os import path

    net = NNet('best')

    examples = []   #[board, pi, winner] moves of played games

    while True:

        # Update chunk counter
        with chunk_counter.get_lock():
            chunk_counter.value -= 1
            chunk_index = chunk_counter.value

        if chunk_index < 0:
            break

        print('--', cp().name, '-- Starting chunk', chunk_index)

        for game_it in range(games_per_chunk*chunk_index, games_per_chunk*(chunk_index+1)):

            game_moves = []  #[board, pi, winner] moves of single game

            # Init game environment
            game, first_player = GameState.generate_training_game(game_it)
            mcts = MCTS(net, game.copy())

            # Game loop
            player = first_player
            while not game.is_finished():
                pi = mcts.get_pi(player)
                if train_args.show_moves:
                    print(str(game)+'\n\n\n')
                    sleep(1)
                assert(not any(np.isnan(pi))) #debug only
                game_moves.append([game.canon_board(player), pi, 0])   #winner is set temporary to 0
                game.next_state(np.argmax(pi), player)
                player *= -1

            print('--', cp().name, '-- selfplay match ended')

            # Iterate over game_moves to set the right winner after the game
            winner = game.get_winner()
            if winner:   #winner==0: game is tie
                winner = 1 if winner==first_player else -1
                for move in game_moves:
                    move[2] = winner
                    winner *= -1
        
            """
            # Discard first moves
            if len(game_moves) >= 8:
                game_moves = game_moves[8:]
            """

            # Add moves of this game if it's not a tie
            #if not winner==0:
            examples += game_moves
            
        tmp_dir = 'tmp/'
        if not path.exists(tmp_dir):
            mkdir(tmp_dir)
        with open(tmp_dir+'examples_{}.pickle'.format(cp().name), 'wb') as f:
            dump([chunk_index, examples], f)


    print('--', cp().name, '-- Finished')
    
    # Append moves to the queue
    all_moves.put(examples)

    return 0








def train_net(examples):
    """
    Call a single external process to train the net
    """
    p = mp.Process(target=train_net_process, args=(examples,))
    p.start()
    p.join()
    #input('Joined train process. ')

def train_net_process(examples):
    globals.set_gpu_visible(True)
    from NNet import NNet
    net = NNet('best')
    net.train(examples)
    net.save(globals.get_last_iteration()+1)  #save after training as <iteration+1>'.h5'
    print('Saved trained net in {}.h5 file'.format(globals.get_last_iteration()+1))






def evaluate_net(net=None, net_opponent=None):
    """
    Evaluates net against net_opponent. By default evaluates 'best' net against
    the recently trained one
    """
    if not net_opponent or not net:
        raise ValueError('nets can\'t be None')

    chunk_total   = ceil(train_args.num_eval_games / train_args.games_per_chunk)
    chunk_counter = mp.Value(c_int, chunk_total)  #decremental counter
    results_queue = mp.Queue()

    processes = [None] * train_args.parallel_num

    #p = mp.Process(target=train_net_process, args=(examples,))
    for i in range(train_args.parallel_num):
        processes[i] = mp.Process(target=evaluate_net_process, 
            args=(net, net_opponent, chunk_counter, train_args.games_per_chunk, results_queue,))
        processes[i].start()
    
    results = 0
    for p in processes:
        #p.join()  NO! It deadlocks
        results += results_queue.get()
    
    results_queue.close()

    #print(np.asarray(all_moves).shape)   #debug purposes, should be (x, 3)
    return results


def evaluate_net_process(net, net_opponent, chunk_counter, games_per_chunk, results_queue):
    """
    Best network plays some games against the new trained one.
    Returns the sum of ties, wins and loses, where ties are 0,
    wins are 1 and loses -1.
    """
    globals.set_gpu_visible(False)
    from multiprocessing import current_process as cp
    from time import sleep
    from MCTS import MCTS
    from NNet import NNet

    net = NNet(net or (globals.get_last_iteration()+1))
    net_opponent = NNet(net_opponent)

    result = 0

    while True:

        # Update chunk counter
        with chunk_counter.get_lock():
            chunk_counter.value -= 1
            chunk_index = chunk_counter.value

        if chunk_index < 0:
            break

        print('--', cp().name, '-- Starting chunk', chunk_index)

        for game_it in range(games_per_chunk*chunk_index, games_per_chunk*(chunk_index+1)):

            # init environment
            game, first_player = GameState.generate_training_game(game_it)
            mcts = MCTS(net, game.copy())
            mcts_opponent = MCTS(net_opponent, game.copy())

            player = first_player
            # Game loop
            while True:
                # debug
                #print('root state', mcts.root.state)

                #net turn
                action = np.argmax(mcts.get_pi(player))
                game.next_state(action, player)
                if train_args.show_moves:
                    print('\n\n\nnet MOVED\n'+str(game.board))
                    sleep(0.7)
                
                if game.is_finished():
                    break

                if not mcts_opponent.root.has_children():
                    #if there are no children, expand the current mcts root, but 
                    #first check if the game is finished. If that's the case nothing
                    #is expanded
                    mcts_opponent.search(mcts_opponent.root, player)
                assert(mcts_opponent.root.children[action])
                mcts_opponent.root = mcts_opponent.root.children[action]

                #net_opponent turn
                # debug
                #print('root state', mcts_opponent.root.state)

                action = np.argmax(mcts_opponent.get_pi(-player))
                game.next_state(action, -player)
                if train_args.show_moves:
                    print('\n\n\nnet_opponent MOVED\n'+str(game.board))
                    sleep(0.7)

                if game.is_finished():
                    break

                if not mcts.root.has_children():
                    #same as above
                    mcts.search(mcts.root, -player)
                assert(mcts.root.children[action])
                mcts.root = mcts.root.children[action]

            #new net was the first to move
            result += 0 if not any(game.valid_moves()) else (1 if game.get_winner()==first_player else -1)

            print('--', cp().name, '-- selfplay match ended')
            #print('.', end='')

    print('--', cp().name, '-- Finished')

    results_queue.put(result)

    return 0



if __name__ == '__main__':
    if sys.argv[-1] == '-c':
        print('Executing custom snippet')
        train_args.games_per_chunk = 3
        scores = []
        ev = lambda n1,n2,s: s.append((n1,n2,evaluate_net(net=n1,net_opponent=n2)))
        ev('53', '54', scores)
        print(scores)
        ev('53', '55', scores)
        print(scores)
        ev('54', '53', scores)
        print(scores)
        ev('55', '53', scores)
        print(scores)
        ev('55', '54', scores)
        print(scores)
        ev('54', '55', scores)
        pprint(scores)

    else:
        if not train_args.evaluate_only:
            if train_args.show_moves:
                train_args.parallel_num = 1
            train(good_luck=True)
        else:
            if train_args.show_moves:
                train_args.parallel_num = 1
            print('Evaluating')
            s1 = evaluate_net(net='1', net_opponent='best')
            print('Score:', s1)

