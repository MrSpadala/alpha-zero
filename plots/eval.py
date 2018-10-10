from NNet import NNet
from pickle import load
import sys
'''
it = sys.argv[-1]
try:
  int(it)
except:
  print('pass iteration')
  raise Exception
'''

its = [1,3,5,7,10,13,15,17,20,23,25,27,30,
33,35,37,40,43,45,47,50,53,55,57,60]
#its = [60]
for it in its:
  net = NNet(str(it))
  res = net.evaluate(load(open('selfplay_data/examples_it_60.pickle', 'rb')))
  print(it, res)
