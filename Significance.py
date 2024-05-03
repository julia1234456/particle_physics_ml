import ROOT 
import math
import numpy as np
import csv
from enum import Enum
import sys
import matplotlib.pyplot as plt
import csv
import logging
import matplotlib as mpl
import pandas as pd
from collections import defaultdict
columns = defaultdict(list)
ROOT.gROOT.SetBatch(True)

def create_histo( hists, samples, treename, cut):

    h = {sample: {hist: None for hist in hists.keys()} for sample in samples.keys()}

    for sample in samples.keys() : 
        for hist in hists.keys() :
    
            file = ROOT.TFile(samples[sample], "READ")
            tree = file.Get(treename)
    
            #Create histogram 
            hist_name = '%s_%s' % (sample, hist)
            hist_title = '%s for %s sample' % (hist, sample )
            hist_param = hists.get(hist)
            histo = ROOT.TH1F(hist_name, hist_title, hist_param[0], hist_param[1], hist_param[2])
    
            # Fill histogram
            tree.Draw('%s>>%s' % (hist, hist_name),cut, 'goff')
            h[sample][hist] = histo.Clone()
            h[sample][hist].SetDirectory(0) 
    
    # Create the merged histogram (addition all backgrounds)
    assert(len(h) > 1)
    h['merge']  = { hist : None for hist in hists.keys()}
    for hist in hists.keys():
        hist_param = hists.get(hist)
        hm = ROOT.TH1F('merge', 'merge', hist_param[0], hist_param[1], hist_param[2])
        for sample in samples.keys():
            if sample != 'sgn':
                hm.Add(h[sample][hist], 1)
        h['merge'][hist] = hm.Clone() 
        h['merge'][hist].SetDirectory(0)
        del hm

   
            
    return h 
      




  
