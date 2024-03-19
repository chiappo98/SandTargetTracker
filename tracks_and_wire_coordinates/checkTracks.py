from matplotlib import pyplot as plt
import numpy as np
import argparse
import json
from hitPreprocessing import Tracks

def fitTrack(hitFile, channel):
    t = Tracks(hitFile, channel, 0.1e-10)
    t.import2RDF()    
    return t.proj_x, t.proj_y, t.tan_x, t.tan_y, t.x, t.y, t.sx, t.sy

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    #parser.add_argument('measureFile', type=str, help='.json file with computed wire coordinates')
    args = parser.parse_args()
    
    px, py, tx, ty, x, y, sx, sy = fitTrack(args.rootFile, "dc1") 

    plt.figure("original")
    plt.plot(x+sx*25, y+sy*25, '.', markersize='2')    
    plt.figure("new")
    plt.plot(px+tx*25, py+ty*25, '.', markersize='2')  
    plt.figure("comparison")
    plt.plot(x+sx*25, y+sy*25, '.', markersize='2')  
    plt.plot((px+tx*25)[((tx<0.3) & (tx>-0.3)) & ((ty<0.3) & (ty>-0.3))], (py+ty*25)[((tx<0.3) & (tx>-0.3)) & ((ty<0.3) & (ty>-0.3))], '.', markersize='2') 
    plt.show()