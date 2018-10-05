
import numpy            as np
from   os               import mkdir, path
from   keras.models     import *
from   keras.layers     import *
from   keras.optimizers import Adam

from game    import GameState
from globals import *


"""
This module contains the neural network and some utility methods
"""


nnet_args = dotdict({
    'lr' : 0.0019,
    'batch_size' : 64,
    'epochs' : 18,
    #'epochs' : 1,

    #hyperparameters to NOT TO CHANGE during training
    'filters' : 256,
    'pi_filters' : 2,
    'v_filters'  : 1,
    'hidden_dense' : 256,


})




class NNet():

    def __init__(self, model):
        if model is None: #create new model
            # init
            garbage = GameState()
            shape = garbage.board.shape
            output_size = garbage.action_size()
            del garbage
            
            # Neural net
            input_board = Input(shape=shape)
            input_board_reshaped = Reshape((shape[0], shape[1], 1))(input_board)

            conv1 = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.filters, (4,4), padding='same')(input_board_reshaped)))
            conv2 = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.filters, (3,3), padding='same')(conv1)))
            conv3 = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.filters, (3,3), padding='same')(conv2)))
            conv4 = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.filters, (3,3), padding='same')(conv3)))
            
            #conv4 -> pi
            conv_pi = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.pi_filters, (1,1))(conv4)))
            conv_pi_flat = Flatten()(conv_pi)
            pi = Dense(output_size, activation='softmax', name='pi')(conv_pi_flat)

            #conv4 -> v
            conv_v = Activation('relu')(BatchNormalization(axis=3)
                (Conv2D(nnet_args.v_filters, (1,1))(conv4)))
            conv_v_flat = Flatten()(conv_v)
            hidden_v = Dense(nnet_args.hidden_dense, activation='relu')(conv_v_flat)
            v = Dense(1, activation='sigmoid', name='v')(hidden_v)
            
            self.model = Model(inputs = input_board, outputs=[pi,v])
            self.model.compile(loss=['categorical_crossentropy','mean_squared_error'],
                optimizer=Adam(nnet_args.lr))   #maybe todo loss function

        else:
            self.model = load_model(global_args.save_dir+str(model)+'.h5')



    def train(self, examples, validation_modifier=0.0, epochs=None):
        states, pis, vs = list(zip(*examples))  #pis and vs will be our targets
        states, pis, vs = np.asarray(states), np.asarray(pis), np.asarray(vs)
        self.model.fit(x=states, y=[pis, vs], batch_size=nnet_args.batch_size, 
            validation_split=validation_modifier, epochs=epochs or nnet_args.epochs)


        

    def save(self, iteration):
        self.model.save(global_args.save_dir+str(iteration)+'.h5')


    def predict(self, board):
        """
        Returns (pi, v)
        """
        pi, v = self.model.predict(np.asarray([board]))

        #v[0] is an array of length 1

        # Normalize sigmoid activation function of 'v' over [-1, 1]
        v_norm = v[0][0]*2 -1


        return pi[0], v_norm

