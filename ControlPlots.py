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
    """
    Create merge histograms to compare the signal distribution 
    to different background distributions for different parameters.

    hists: List of parameters.
    samples : List of signal and background root files
    outpath : Path to the folder storing the output histograms.
    Normalize : if True, create a normalized graph. 

    """

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


if __name__ == "__main__":

    # Modify basepath and outpath according to your environment.
    basepath = '/home/julia/TMVA/'
    outpath = '/plotsCompare/Compare/'
    treename = 'coupl'

    # List of parameters in root files for which we want to draw the distribution. 

    hists = {
        'n_Jets': [10, 0, 8, 'Jet multiplicity'],
        'bb_Mass': [50, 0, 220, 'M_{bb} [GeV]'],
        'bbmiss_Mass': [50, 30, 300, 'M_{bbE^{miss}_{T}} [GeV]'],
        'bb_pT': [50, 0, 160, 'p_{T}^{bb} [GeV]'],
        'bb_Et': [50, 0, 220, 'E_{T}^{bb} [GeV]'],
        'bb_dR': [50, 0, 5, '#Delta R_{bb}' ],
        'RatioJet1': [50, 0, 2.5, 'RatioJet1'],
        'RatioJet2': [50, 0, 2.5, 'RatioJet2'],
        'met_Met': [50, 0, 120, 'E^{miss}_{T} [GeV]'],
        'leadpT_Jet_Eta': [50, -2.5, -2.5, '#eta_{b1}'],
        'subleadpT_Jet_Eta': [50, -2.5, -2.5, '#eta_{b2}'],
        'leadpT_Jet': [50, 0, 120, 'p_{T}^{b1} [GeV]'],
        'subleadpT_Jet': [50, 0, 80, 'p_{T}^{b2} [GeV]'],
        'BDT_response': [50, -0.4, 0.4,'BDT score' ],
        'BDTG_response': [50, -1, 1.2, 'BDTG score']
    }
    


    # List of root files for signal and background samples.
    samples = {
        'sgn': basepath + '2b2vl_sgn_BDT.root',    
        'bbjj': basepath + 'bbjj_bkg_BDT.root',
        'zz': basepath + 'zz_bkg_BDT.root',
        'ww': basepath + 'ww_bkg_BDT.root'
    }
    
    # Create merge histograms comparing signal distribution to different background distribution.
    create_histo(hists, samples,outpath, True)
        
     












