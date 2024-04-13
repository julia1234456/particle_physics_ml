import ROOT
ROOT.gROOT.SetBatch(True)

import os
cwd = os.getcwd()
stylepath = cwd + '/AtlasStyle/'
print(stylepath)

ROOT.gROOT.LoadMacro( stylepath + "AtlasStyle.C")
ROOT.gROOT.LoadMacro( stylepath + "AtlasLabels.C")
ROOT.gROOT.LoadMacro( stylepath + "AtlasUtils.C")
ROOT.SetAtlasStyle()
from ROOT import *


def create_histo(hists, samples, outpath, Normalize):

    h = {sample: {hist: None for hist in hists.keys()} for sample in samples.keys()}

    for sample in samples.keys() : 
        for hist in hists.keys() : 
            
            file = ROOT.TFile(samples[sample], "READ")
            tree = file.Get(treename)

            # Create histogram 
            hist_name = '%s_%s' % (sample, hist )
            hist_title = sample
            hist_param = hists.get(hist)
            histo = ROOT.TH1F(hist_name, hist_title, hist_param[0], hist_param[1], hist_param[2])
            
            # Fill histogram 
            tree.Draw('%s>>%s' % (hist, hist_name),'(BDTG_response > -0.1425)*1','goff')
            if Normalize == True :
                scale = 1/histo.Integral()
                histo.Scale(scale)
        
            h[sample][hist] = histo.Clone()
            h[sample][hist].SetDirectory(0)       
        
     












