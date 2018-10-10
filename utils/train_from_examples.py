
import sys
sys.path.append('../src/')
import pickle
import train

if len(sys.argv) != 4:
    print('Usage:',argv[0],'<checkpoint to load> <examples to train on> <checkpoint to save>')

examples = pickle.load(open(argv[2], 'rb'))

from NNet import NNet
net = NNet(argv[1])
net.train(examples, validation_modifier=0.2, epochs=18)
net.save(argv[3])

#print('Saved trained net in {}.h5 file'.format(globals.get_last_iteration()+1))




