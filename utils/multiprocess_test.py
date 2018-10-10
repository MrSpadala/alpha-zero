
"""
Some testing when developing multiprocess architecture
"""


# TEST RESULT: same speed, maybe better with one queue and a single lock



# DOUBLE QUEUE, SINGLE LOCK

from multiprocessing import Pool, Lock, Process, Queue, Manager


def net_worker(q_req, q_res, tasks):
	try:
		_net_worker(q_req, q_res, tasks)
		print('BLBLBL')
	except EOFError:
		print('Finished')


def _net_worker(q_req, q_res, tasks):
	from NNet import NNet
	import numpy as np
	from time import time

	t = tasks
	
	net1 = NNet('best')
	net2 = NNet(1)
	while t > 0:
		b = q_req.get(block=True)
		if b is None:
			t -= 1
			continue

		s = time()
		res = net1.predict(b)
		manz = net2.predict(b)
		e = time()
		q_res.put((res,manz))
		print(e-s)


def worker(args):
	import numpy as np
	from multiprocessing import current_process as cp
	from time import time
	q_req, q_res, lock = args
	for i in range(20):
		b = np.random.rand(6,7)*2 -1
		lock.acquire()
		
		q_req.put(b)
		q_res.get()
		
		lock.release()
	q_req.put(None)


m = Manager()
q_req = m.Queue(1)
q_res = m.Queue(1)
l = m.Lock()

tasks = 6

net = Process(target=net_worker, args=(q_req, q_res, tasks, )).start()

from time import sleep
sleep(4)

with Pool(4) as pool:
	pool.map(worker, [(q_req, q_res, l)]*tasks)

"""


#SINGLE QUEUE, DOUBLE LOCK


from multiprocessing import Pool, Lock, Process, Manager
import multiprocessing as mp


def net_worker(q_req, q_res):
	try:
		_net_worker(q_req, q_res)
	except EOFError:
		print('Finished')



def _net_worker(q, l_queue):
	from NNet import NNet
	import numpy as np
	
	net = NNet('best')
	while True:
		b = q.get(block=True)
		
		res = net.predict(b)
		
		q.put(res)
		l_queue.acquire()
		l_queue.release()


def worker(args):
	import numpy as np
	from multiprocessing import current_process as cp
	from time import time
	q, l_worker, l_queue = args
	for i in range(20):
		b = np.random.rand(6,7)*2 -1
		a=0
		#for i in range(1000000):
		for manz in range(1):
			a+=1
		l_worker.acquire()
		l_queue.acquire()
		q.put(b)
		q.get()
		l_queue.release()		
		l_worker.release()
		print(cp(), i)


m = Manager()
q = m.Queue(1)
l_worker = m.Lock()
l_queue = m.Lock()

net = Process(target=net_worker, args=(mp.Queue(), l_queue,)).start()

from time import sleep
sleep(4)

with Pool(4) as pool:
	pool.map(worker, [(q, l_worker, l_queue)]*6)

"""
