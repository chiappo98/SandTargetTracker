from matplotlib import pyplot as plt
import numpy as np
import argparse
from hitPreprocessing import Tracks

def fitTrack(hitFile, channel, method):
    t = Tracks(hitFile, channel, 0.1e-10)
    if method=="all_wires":
        t.import2RDF(method)    
        return 0,0,0,0,0,0,0,0,0,0
    elif method=="single_wire":
        t.import2RDF(method)    
        return t.proj_x, t.proj_y, t.tan_x, t.tan_y, t.x, t.y, t.sx, t.sy, t.ix, t.iy

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('rootFile', type=str, help='.root file with hits')
    parser.add_argument('method', type=str, help='"all_wires or single_wire modes"')
    args = parser.parse_args()
    
    px, py, tx, ty, x, y, sx, sy, ix, iy = fitTrack(args.rootFile, "dc0_1", args.method) 

    if px[0]!=0:
        filt = ((tx<0.3) & (tx>-0.3)) & ((ty<0.3) & (ty>-0.3))
        plt.figure("original")
        plt.plot(x+sx*25, y+sy*25, '.', markersize='2')    
        plt.figure("new")
        plt.plot(px+tx*25, py+ty*25, '.', markersize='2')  
        plt.figure("comparison")
        plt.plot(x+sx*25, y+sy*25, '.', markersize='2')  
        plt.plot((px+tx*25)[filt], (py+ty*25)[filt], '.', markersize='2') 
        plt.show()