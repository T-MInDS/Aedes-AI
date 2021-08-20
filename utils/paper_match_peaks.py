import numpy as np, pandas as pd
import os, argparse, json, pdb
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict
#from match_peaks import *

tests=glob('./results/Test/*')
trains=glob('./results/Train/*')
vals=glob('./results/Val/*')

files=[*trains, *vals, *tests]
#files=trains

for file in files:
    print(file)
    os.system("python utils/match_peaks.py -r "+file)
