
from platform import system
from shutil   import copy   as cp
from os       import remove as rm, path, mkdir

"""
This module contains global variables and $hit used by other modules
"""

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


global_args = dotdict({
    'DEBUG' : True,

    #need the final slash or backslash
    'save_dir' : 'checkpoints'+('\\' if system()=='Windows' else '/'),

    #self-play data save folder
    'selfplay_dir' : 'selfplay_data'+('\\' if system()=='Windows' else '/'),

    #files that keep track of the number of iterations done
    'it_file' : 'iteration.number',
})




def get_last_iteration():
    """
    Search for a file in the current directory
    """
    fname = global_args.it_file
    with open(fname, 'r') as f:
        iteration = int(f.readline()[:-1])  #[:-1] to chop the newline
    return iteration



_file_text = """
##############
This file is used to keep track of the current iteration number
"""

def update_iteration_file(it):
    with open(global_args.it_file, 'w') as f:
        f.write(str(it)+'\n')
        f.write(_file_text)


def init_training_stuff():
    """
    Check if all files and folders are ok
    """
    #create checkpoint folder
    if not path.exists(global_args.save_dir):
        print('Creating checkpoint folder...')
        mkdir(global_args.save_dir)
    #check if checkpoint file exists
    if not path.exists(global_args.it_file):
        print(('Checkpoint file {} not found in current directory, if you continue the training '+
            'will be restarted from iteration 0').format(global_args.it_file))
        input('(ENTER to continue, CTRL-C to stop) ')
        update_iteration_file(0)
    #check if checkpoint file is well formed
    try:
        get_last_iteration()
    except ValueError:
        print(('Checkpoint file {} is corrupted. Make sure the first line of '+
            'the file contains the number of iterations done. If you continue '+
            'the training will be restarted from iteration 0').format(global_args.it_file))
        input('(ENTER to continue, CTRL-C to stop) ')
        update_iteration_file(0)
    #if there is not 'best.h5' model saved then create a new one
    if not path.exists(global_args.save_dir+'best.h5'):
        from multiprocessing import Process
        print('Creating a new best.h5 file...')
        def init_net_model():
            from NNet import NNet
            net = NNet(None)
            net.save('best')
        p = Process(target=init_net_model)
        p.start()
        p.join()


def set_gpu_visible(flag):
    """
    To call before importing keras or tensorflow. You may ask why do you have to hide
    your gpu. Well i did my testings with a Gigabyte GTX 780 and an i5-4670@3.4GHz:
    to use the gpu, you can't simply launch 4 process in parallel each importing its
    own copy of tensorflow, keras and the model of the net, because the VRAM saturates
    pretty fast. So what i came up with was a single process running keras with the
    GPU and other 4 processes running Monte Carlo simulations. These process will
    enqueue requests to the keras process and it will return the output of the net.
    Results were that a single GPU bottlenecked a lot, resulting in an average time
    (from the instant that a process enqueued the request to the moment it received
    results) of 0.015s. For comparison using cpu only on 4 process takes 0.0055s.
    Even reducing the number of parallel process didn't help, resulting in a 0.008s
    average 
    """
    import os
    import tensorflow as tf
    from keras import backend as K
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config  = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 1 if flag else 0})
    session = tf.Session(config=config)
    K.set_session(session)



# - - - Utilities - - - #

def accept_model():
    folder = global_args.save_dir
    rm(folder+'best.h5')
    cp(folder+str(get_last_iteration()+1)+'.h5', folder+'best.h5')

def dump_selfplay_data(examples):
    from pickle import dump
    d = global_args.selfplay_dir
    if not path.exists(d):
        print('Creating selfplay data folder...')
        mkdir(global_args.selfplay_dir)
    with open(d+'examples_it_{}.pickle'.format(get_last_iteration()), 'wb') as f:
        dump(examples, f)







