
import sys
sys.path += ['src/', 'utils/']

import human_play


if len(sys.argv) == 1:
    raise Exception('Usage:',sys.argv[0],'train|play')

if sys.argv[1] == 'play':
    if len(sys.argv) < 3:
        print('Usage:',sys.argv[0],'train <net checkpoint>')
        print('Using default net checkpoint \'best\'')
        net = 'best'
    else:
        net = sys.argv[2]
    human_play.main(net)
