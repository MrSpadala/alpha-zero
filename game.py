
import numpy  as np
from   math   import log, ceil, floor
from   random import randint

from globals import *

"""
This module implements the logic of connect four
"""

class GameState():
    def __init__(self):
        self.cols, self.rows = 7, 6
        self.board = np.zeros((self.rows, self.cols))
        self.last_action = None  #(row, col) of last action performed
    
    def action_size(self):
        """Returns the number of all possible actions"""
        return self.cols
    
    def valid_moves(self):
        """
        Returns an array of length action_size, and its value[i]
        it's 0 if the move 'i' is not valid.
        """
        return abs(self.board[0])-1

    def canon_board(self, player):
        """Returns the board from the point of view of player 1"""
        assert(abs(player)==1)
        return player * self.board

    def copy(self):
        gs = GameState()
        gs.board, gs.last_action = self.board.copy(), self.last_action
        return gs

    def get_winner(self):
        """
        Returns 1 or -1 if either player 1 or -1 is the 
        winner of the game. If no one won OR the game is
        over with a tie, then it returns 0. To check if
        a game is tie you should call valid_moves: if there
        are no valid moves then the game is tie. In other words 
          not any(valid_moves(board))  ==  {game is tie}

        """
        if self.rows!=6 or self.cols!=7:
            raise ValueError('get_winner rules written only for 6x7 boards')

        if not self.last_action:
            return 0

        #col_index = self.last_action[1]
        #row_index = np.nonzero(self.board[:,col_index])[0] #first non zero element in the column of 'last_action'
        row_index, col_index = self.last_action
        row   = self.board[row_index, :]
        col   = self.board[:, col_index]
        diag1 = np.diagonal(self.board, col_index-row_index)
        diag2 = np.diagonal(np.fliplr(self.board), -(col_index+row_index-6))

        for line in [row, col, diag1, diag2]:
            if line.shape[0] < 4:
                continue

            for four in [line[i:i+4] for i in range(len(line)-3)]:
                if sum(four) == 4:
                    return 1
                elif sum(four) == -4:
                    return -1
        return 0

    def is_finished(self):
        return self.get_winner() or not any(self.valid_moves())
    
    def next_state(self, action, player, copy=False):
        """
        Performs the action by the player.
        If copy is set then a new GameState object is returned.
        Otherwise is returned None
        """
        if global_args.DEBUG:
            assert(abs(player)==1)
            s = np.sum(self.board)
            #assert(abs(s)<=1)
            assert(s==int(s))
            assert(self.valid_moves()[action])
            assert(not self.is_finished())
        
        gs = self if not copy else self.copy()
        first_non_zero = np.nonzero(gs.board[:,action])[0]
        row = first_non_zero[0]-1 if len(first_non_zero) else self.rows-1
        gs.board[row][action] = player
        gs.last_action = (row, action)
        return gs if copy else None

    @staticmethod
    def all_symmetries(examples):
        """
        Given a list of examples in the form [state, pi, v], returns
        an extended list with symmetries of the board, in this 
        case symmetry along the y-axis for both state and pi,
        the v remains unchanged
        """
        symmetries = map(lambda e: (np.fliplr(e[0]), np.flip(e[1],0), e[2]), examples)
        res = list(examples) + list(symmetries)
        return res


    @staticmethod
    def generate_training_game(iteration):
        gs, player = None, None
        randomizer = randint(0, 10000)
        i = 0
        while (gs is None) or (gs.is_finished()):
            gs, player = GameState._generate_training_game(iteration+i*randomizer)
            i += 1
        return gs, player


    @staticmethod
    def _generate_training_game(iteration):
        """
        This function returns a board with two action performed. Those 
        two actions depend on the iteration number It's used during 
        self play, to generate all possible boards with two moves.
        It also returns the next player which has to move
        """
        gs = GameState()
        size = gs.action_size()
        #if iteration >= size**2:
        #iteration = randint(size**2, size**4)

        #get the seed
        if iteration > 7**3:
            iteration = (hash(str(iteration*1000)) ** 2) % 7**6

        moves = []
        player = 1
        if iteration == 0:
            gs.next_state(0, player)
            return gs, -player
        while iteration > 0:
            moves.append(iteration % size)
            iteration = iteration // size
            
        for i in reversed(range(len(moves))):
            if i == len(moves)-1 and len(moves)>1:
                moves[i] -= 1
            gs.next_state(moves[i], player)
            player *= -1
        return gs, player

    def __str__(self):
        s = ''
        for line in self.board:
            for n in line:
                if n==0:
                    s += '  '
                elif n==1:
                    s += 'X '
                else:
                    s += 'O '
            s += '\n'
        return s
    def __repr__(self):
        return str(self.board)


'''if __name__ == '__main__':
    #= int(input(': '))
    for n in range(50):
        print(generate_training_board(n))'''




    
