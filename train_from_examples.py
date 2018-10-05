
import pickle
import train

examples = pickle.load(open('selfplay_data/examples_it_54.pickle', 'rb'))

from NNet import NNet
net = NNet('54')
net.train(examples, validation_modifier=0.0, epochs=18)
net.save('55_retrained')

#print('Saved trained net in {}.h5 file'.format(globals.get_last_iteration()+1))




