"""
Data IO routines specially for mem sharing purpose
using joblib library
"""

from sklearn.externals import joblib
from os import path
from multicore import MulticoreJob

def persist_named_data(folder, named_data):
    """usually used in one process (machine) only
    """
    data_files = []
    for (name, data) in named_data.items():
        data_file = path.abspath(path.join(folder, name + '.pkl'))
        joblib.dump(data, data_file)
        data_files.append((name, data_file))
    return data_files
    
def load_named_data(data_files):
    """usually used in multiple processes
    """
    from sklearn.externals import joblib
    named_data = {name : joblib.load(data_file) for (name, data_file) in data_files}
    return named_data

def test():
    ## test persist_named_data
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    split = X.shape[0] / 2
    named_data = {'part1': (X[:split], y[:split]), 'part2': (X[split:], y[split:])}
    data_files = persist_named_data('../tmp/', named_data)
    print data_files
    ## test load_named_data
    jobber = MulticoreJob()
    jobber.apply(load_named_data, {'data_files':data_files})
    while not jobber.isready():
        print jobber.progress()
    print jobber.partial_result()
    print 'all tests passed...'
    
if __name__ == '__main__':
    test()