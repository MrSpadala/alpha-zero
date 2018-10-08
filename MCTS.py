
import numpy     as np
from   math      import sqrt

from game    import *
from globals import *


"""
This module implements the monte carlo tree search guided
by a neural network
The tree is kept in memory by saving the root edge, and then
every edge has a list of children edges. (Read below) 
"""


mcts_args = dotdict({
    #numbers of mcts simulation for each move (1600 in alphago)
    'sim_per_move' : 1400,  

    #hyperparameter determining exploitation/exploration during
    #the tree search.
    #cpupct -> 1 preferes exploration. 
    'cpuct' : 1.9,

    #hyperparameter determining the pi returned by the mcts. Let be 'a'
    #the best action: if tau is zero, then the pi returned will be all
    #zeros except for the 'a' component which will be set to 1. Instead,
    #when tau>0 all probabilties gets into the pi vector
    'tau' : 0.7,  #0<=tau<=1

    #number of moves before switching tau
    'tau_moves' : 14,

    #artificial noise multiplier, added to the returning pi.
    #  pi += noise * noise_multiplier
    #noise is a random probability distribution, sum(noise)==1
    'noise_multiplier' : 0.6,

    #if true prints mcts stats for every chosen move
    'show_moves_detail': False
})




class MCTS:

    def __init__(self, net_model, state):
        self.net  = net_model
        self.root = self.Edge(state, 0)
        #tau
        self.tau = mcts_args.tau
        self._move_count = 0


    class Edge:
        """
        The tree is represented in memory as a tree of edges,
        where every edge has an attribute self.children which is
        list of all its children. If an edge hasn't been exapnded
        yet by the net, then its self.children will be None.
        """

        def __init__(self, state, p):
            self.state = state
            self.N = 0  #visit count
            self.W = 0  #total action value
            self.Q = 0  #mean action value
            self.P = p  #probability of selecting this edge
            self.children = None
            self.valids = state.valid_moves()

        def __repr__(self):
            s = 'EDGE BOARD:\n'
            s += self.state.__repr__()+'\n'
            s += str(self.N)+'  '+str(self.W)+'  '+str(self.Q)+'  '+str(self.P)+'\n'
            s += 'children is none:'+str(self.children==None)+'\n'
            return s

        def has_children(self):
            #Just one utility method for Edge class
            return not self.children is None



    def get_pi(self, player):
        """
        Returns the best and valid action found accordingly to the tree search.
        THEN (before returning the action) it moves down the tree so that the action
        returned will be the root of the next tree search (useful during selfplays).
        When calling get_pi make sure that the game isn't finished.

        Returns:
            pi: vector of size action_size() with each component pi[i] is the
                probability of the action 'i'. If p[i] is zero, then the action 'i'
                is not a valid move
        """
        for sim in range(mcts_args.sim_per_move):
            assert(self.root)
            self.search(self.root, player)

        visits = [c.N if not c is None else np.nan for c in self.root.children]

        if self._move_count>mcts_args.tau_moves:
            self.tau = 0
        
        visits_norm = np.nan_to_num(np.asarray(visits) / np.nansum(visits))
        
        if mcts_args.noise_multiplier > 0:
            noise = np.random.random(visits_norm.shape)
            noise[(visits_norm==0)] = 0
            noise_norm = noise / np.sum(noise)
            visits_norm += noise_norm * mcts_args.noise_multiplier
            visits_norm /= np.sum(visits_norm) 


        # Get pi using temperature (tau) parameter
        if self.tau<=0:  #tau should be always >=0
            pi = np.zeros(self.root.state.action_size())
            pi[np.argmax(visits_norm)] = 1
        else:
            pi = visits_norm ** (1/self.tau)
            pi /= sum(pi)

        # MCTS data
        if mcts_args.show_moves_detail:
            print('visits', visits)
            print('pi', pi)
            print('prior', [c.P if not c is None else -1 for c in self.root.children])
            print('Q values', [c.Q if not c is None else -1 for c in self.root.children])

        self.root = self.root.children[np.argmax(pi)]  #scale the tree
        self._move_count += 1

        return pi



    
    def search(self, edge, player):
        """Implements the tree search"""
        assert(abs(player)==1)  #debug only

        # Check if game is ended before continuing
        if edge.state.is_finished():
            winner = edge.state.get_winner()
            winner = 0 if not winner else (1 if winner==-player else -1)  #-player cause the winning move was done by -player
            #winner *= 2  #try to give more weigth to the winner
            edge.N += 1
            edge.W += winner 
            edge.Q  = edge.W/edge.N
            return -winner


        # if this edge hasn't children: try to expand and evaluate every valid move using the net,
        # else: select one of its children
        if not edge.has_children():

            #since player can be +1 or -1 and edge.state can be filled only with
            #+1, 0 or -1, multiplying player and state let the net predict correctly
            #for both player +1 and -1
            #s = time()
            pi, v = self.net.predict(edge.state.canon_board(player))
            #e = time()
            #print(e-s)

            # With GPU is WAAAAAY slower

            # These will be our new states with the action performed
            # (next_state function returns a deep copy of the updated state)
            states = [edge.state.next_state(action, player, copy=True) if valid else None
                for action, valid in zip(range(edge.state.action_size()), edge.valids)]
            
            edge.children = [self.Edge(state, p) if valid else None
                for state, p, valid in zip(states, pi, edge.valids)]

            return -v

        else:
            children = edge.children
            valids   = edge.valids   
            # Choosing the action that maximises Q(s,a)+U(s,a) according to the paper.
            # visits_sum is the sqrt of the sum of visit count of all 
            visits_sum = sqrt(sum([c.N if not c is None else 0 for c in children]))

            U_plus_Q = [(mcts_args.cpuct*children[a].P*visits_sum/(1+children[a].N) + children[a].Q)
                if valid else np.nan for a, valid in zip(range(edge.state.action_size()), valids)]

            action = np.nanargmax(U_plus_Q)

            # debug
            #print('U_plus_Q inside search:', U_plus_Q, visits_sum)
            #print('children', children)
            #print('VISIT', visits_sum, [c.N if c else None for c in children])
            #print('U_plus_Q', U_plus_Q)


            # children[action] is the root where to apply the new recursive search
            assert(children[action] != None)
            
            # Apply search on the selected action
            v = self.search(children[action], -player) #-player because the next search is from the point of view of the other player 

            # Backup values after the recursive call
            edge.N += 1
            edge.W += v
            edge.Q  = edge.W/edge.N
            return -v
            









