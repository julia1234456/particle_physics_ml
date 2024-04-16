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

    # Create merge histograms
    for hist in hists.keys() :
        hs = ROOT.THStack("hs","hs")
        hist_param = hists.get(hist)
        gStyle.SetPalette(kBird)
        canvas =  ROOT.TCanvas('c', 'c', 800, 800)
        
        # Tex in canvas
        tex = ROOT.TLatex(0.2, 0.9, "#sqrt{s} = 250 GeV, L = 250 fb^{-1}")
        tex.SetName('tex')
        tex.SetNDC(True)
        tex.SetTextSize(0.025)
        tex.SetTextColor(kRed)

        # Set maximum Y_axis
        max_global = 0
        for sample in samples.keys():
            max_hist = h[sample][hist].GetMaximum()
            max_global = max_global if max_global > max_hist else max_hist

        # Draw diagrams
        for sample in samples.keys():
            h[sample][hist].SetMaximum(max_global)   
            hs.Add(h[sample][hist])
        hs.SetMaximum(max_global * 4)
        hs.Draw('HISTpfc')
        
     
        # Adjust label and title for x and y axis
        hs.GetXaxis().SetTitle(hist_param[3])
        hs.GetXaxis().SetTitleSize(0.035)  
        hs.GetXaxis().SetLabelSize(0.03)
        if Normalize == True:
            hs.GetYaxis().SetTitle('1 / Events') 
        else : 
            hs.GetYaxis().SetTitle('Events')
        hs.GetYaxis().SetTitleSize(0.035)  
        hs.GetYaxis().SetLabelSize(0.03)
        hs.GetYaxis().SetTitleOffset(1.90)
        
        # Draw Legend
        legend = ROOT.TLegend(0.75, 0.75, 0.90, 0.90) 
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.03) 
        legend.SetShadowColor(0)  
        for sample in samples.keys():
            legend.AddEntry(h[sample][hist], sample, 'f')
        legend.Draw("SAME")

        if Normalize == True:
            canvas.SaveAs( cwd + outpath + '%s_normalize.png' % (hist) )
        else : 
            canvas.SaveAs( cwd + outpath +'%s.png' % (hist) )      
        
     












