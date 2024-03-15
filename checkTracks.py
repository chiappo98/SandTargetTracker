from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from hitPreprocessing import Distance

def fitTrack(hitFile, channel):
    d = Distance(hitFile, channel, 0.1e-10)
    d.import2RDF()    
    # print(len(d.posList))
    return 0

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    #parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    args = parser.parse_args()
    
    fitTrack(args.rootFile, "dc0")