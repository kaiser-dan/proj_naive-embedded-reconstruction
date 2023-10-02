import os

def basenames(files):
    return [os.path.splitext(os.path.basename(filename))[0] for filename in files]

def files_in(dir):
    return os.listdir(dir)