from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from hitPreprocessing import Tracks

def fitTrack(hitFile, channel):
    t = Tracks(hitFile, channel, 0.1e-10)
    t.import2RDF()    
    return 0 #t.proj_y, t.proj_x

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    #parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    args = parser.parse_args()
    
    fitTrack(args.rootFile, "dc0") #proj_y, proj_x = 
    
    #plot projected x and y 