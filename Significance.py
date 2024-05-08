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

def write_csv_significance(outfile_path, xMin, xMax, shape, step, h, label):
    minSel = np.empty(shape)
    rangeSel = np.arange(xMin, xMax, step)
    ind = np.arange(len(minSel))  
    np.put(minSel, ind, rangeSel)

    f = open(outfile_path, 'w+')

    with f:
        writer = csv.writer(f)
        writer.writerow(['BDTscore', 'Nbkg', 'Nsgn', 'Sign'])
    
        for xmin in minSel:
        
            axis = h['sgn'][label].GetXaxis()
            binmin = axis.FindBin(xmin)
            binmax = axis.FindBin(xMax)
            yield_sample = {}
            for hist in h.keys() :
                yield_sample[hist] = h[hist][label].Integral(binmin, binmax)
                
            Nbkg = yield_sample['merge']
            Nsgn = yield_sample['sgn']

            if (Nbkg > 0):
                Sig_G = Nsgn / math.sqrt(Nbkg)
                Sig_P  = math.sqrt( 2*( (Nsgn+Nbkg)*math.log(1+Nsgn/Nbkg)-Nsgn ) )
            else:
                Sig_G = 0
                Sig_P = 0
            print ('BDT: %.4f; Total bkg events: %.5f; Total sgn events: %.5f; Significance_P: %.5f; Significance_P: %.5f'% (xmin, Nbkg, Nsgn, Sig_G, Sig_P))
            writer.writerow(['%.3f' % xmin, '%.8f' % Nbkg, '%.8f' % Nsgn, '%.8f' %Sig_P] )
        
    f.close()
      




  
