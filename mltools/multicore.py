"""
multicore computing package
use command
$ipcluster -n 4 
to enable ipython parallel
"""

import numpy as np

__all__ = []

from IPython.parallel import Client

class MulticoreJob(object):
    def __init__(self):
        self.tasks = {}
        self.client = Client()
        self.lb_view = self.client.load_balanced_view()
    def apply(self, f, named_tasks):
        """named_tasks: dict of {nametask: taskparams}
        """
        self.tasks = { tname:self.lb_view.apply(f, **param) 
                        for (tname, param) in named_tasks.items() }
        return self
    def isready(self):
        return all([t.ready() for t in self.tasks.values()])
    def progress(self):
        return np.mean([t.ready() for t in self.tasks.values()])
    def partial_result(self):
        return {tname:tresult.get() for (tname,tresult) in self.tasks.items()
                    if tresult.ready()}
                    
def test():
    import time
    def gethost():
        import socket, time, random
        time.sleep(random.randint(5, 15))
        return socket.gethostname()
    jobber = MulticoreJob()
    jobber.apply(gethost, dict(zip(xrange(10), [{}]*10))) 
    while not jobber.isready():
        time.sleep(2)  
        print 'progress: ', jobber.progress()
        print jobber.partial_result() 
                       
if __name__ == '__main__':
    test()