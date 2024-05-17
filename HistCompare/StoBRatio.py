import ROOT
import os
from ROOT import *
ROOT.gROOT.SetBatch(True)


cwd = os.getcwd()
stylepath = cwd + '/AtlasStyle/'
print(stylepath)

ROOT.gROOT.LoadMacro( stylepath + "AtlasStyle.C")
ROOT.gROOT.LoadMacro( stylepath + "AtlasLabels.C")
ROOT.gROOT.LoadMacro( stylepath + "AtlasUtils.C")
ROOT.SetAtlasStyle()


def create_hist_ratio(samples, hists, tree_sgn, treename, outfile_path):
    """
    Create the ratio graph between the signal and different background for several parameters. 
        samples : List of background files.
        hists : List of parameters in the signal/background files (such that the mass, the pT..etc) for which we want to draw the ratio graph.
        tree_sgn: Signal tree. 
        treename: Name of the tree in background file (usually same every backgroud and signal). 
    """
    for sample in samples.keys(): 
        if sample != 'sgn':
            for hist in hists.keys():
                hist_param = hists.get(hist)

                file_bkg = ROOT.TFile(samples[sample], "READ")
                tree_bkg = file_bkg.Get(treename)


                histo_sgn = ROOT.TH1F('histo_sgn', 'histo_sgn', hist_param[0], hist_param[1], hist_param[2])
                histo_bkg = ROOT.TH1F('histo_bkg', 'histo_bkg', hist_param[0], hist_param[1], hist_param[2])


                fill_histo(tree_sgn, histo_sgn, hist)
                fill_histo(tree_bkg, histo_bkg, hist)

                normalize(tree_sgn, histo_sgn)
                normalize(tree_bkg, histo_bkg)
                set_maximum(histo_sgn, histo_bkg)

                style_histo_curves(histo_sgn, histo_bkg)
                canvas = ROOT.TCanvas("canvas", "canvas", 800, 600)

                # Create the ratio plot
                rp = ROOT.TRatioPlot(histo_sgn, histo_bkg)
                canvas.SetTicks(0,1)
                rp.Draw('HISTpfc')
                
                # Adjust the x-axis' label and title
                histo_sgn.GetXaxis().SetTitle(hist_param[3])
                rp.GetUpperRefXaxis().SetLabelSize(0.02)
                rp.GetLowerRefXaxis().SetLabelSize(0.02)
                rp.GetLowerRefXaxis().SetTitle(hist_param[3])
                rp.GetLowerRefXaxis().SetTitleOffset(0.95)
                rp.GetLowerRefXaxis().SetTitleSize(0.04)

                # Adjust the upper y-axis' label and title
                rp.GetUpperRefYaxis().SetTitle('1 / NEvents')
                rp.GetUpperRefYaxis().SetLabelSize(0.02) 
                rp.GetUpperRefYaxis().SetTitleOffset(0.95)
                rp.GetUpperRefYaxis().SetTitleSize(0.04)

                # Adjust the lower y-axis' label and title
                rp.GetLowerRefYaxis().SetTitle('Ratio')
                rp.GetLowerRefYaxis().SetTitleOffset(0.95)
                rp.GetLowerRefYaxis().SetTitleSize(0.04)
                rp.GetLowerRefYaxis().SetLabelSize(0.02) 
                

                # Legend 
                legend = ROOT.TLegend(0.70, 0.70, 0.87, 0.90) 
                legend.SetFillColor(0)
                legend.SetFillStyle(0)
                legend.SetTextSize(0.025) 
                legend.SetShadowColor(0)  
                legend.AddEntry(histo_sgn, 'Signal' , 'f')
                legend.AddEntry(histo_bkg, 'Bkg %s' % (sample) , 'l')
                legend.Draw("SAME")
            

                canvas.Update()
                canvas.Modified()
                canvas.SaveAs( cwd + outfile_path + 'signal_%s_%s.png' % (sample, hist))
                del canvas
                              
                file_bkg.Close()
                

def fill_histo(tree, histo, hist): 
    """
    Fill the histogram (histo) with numerical values for the parameter 'hist' stored in the tree. 
    """
    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)
        branch = tree.GetBranch(hist)
        value = branch.GetLeaf(hist).GetValue() 
        histo.Fill(value)

def normalize(tree, histo):
    scale = 1 / tree.GetEntries()
    histo.Scale(scale)

def set_maximum(histo_sgn, histo_bkg): 
    """
    Set a maximum such that the entire curve is displayed on the histogram.
    """
    max_sgn = histo_sgn.GetMaximum()
    max_bkg = histo_bkg.GetMaximum()
    global_max = max_sgn if max_sgn > max_bkg else max_bkg
    histo_sgn.SetMaximum(global_max)
    histo_bkg.SetMaximum(global_max)

def style_histo_curves(histo_sgn, histo_bkg):
    histo_bkg.SetMarkerStyle(kDot)  
    histo_bkg.SetMarkerSize(0.8) 
    histo_bkg.SetLineStyle(1)    
    histo_bkg.SetLineColor(kBlue)
    histo_bkg.SetFillColor(kBlue - 10)

    histo_sgn.SetLineStyle(1)    
    histo_sgn.SetLineColor(kRed)
    histo_sgn.SetFillColor(kRed - 10)
   
if __name__ == "__main__":

    # Modify basepath and outfile_path according to your environment.
    basepath = '/home/julia/TMVA/'
    outfile_path = '/plotsCompare/Ratio/'

    treename = 'coupl'

    samples = {
        'sgn': basepath + '2b2vl_sgn_ntuple.root',
        'bbjj': basepath + '2b2j_bkg_ntuple.root',
        'ww': basepath + 'ww_bkg_ntuple.root',
        'zz': basepath + 'zz_bkg_ntuple.root',
    }

    
    hists = {
        'leadpT_Jet_Mass' : [100, 0, 20, ' Mass [GeV]'],
        'leadpT_Jet': [100, 10, 120, 'pT [GeV]'],
        'met_Met': [100,0,120, 'MET [GeV]']
    }
    
   
  
    file_sgn = ROOT.TFile(samples['sgn'], "READ")
    tree_sgn = file_sgn.Get(treename)
    
    create_hist_ratio(samples, hists, tree_sgn, treename, outfile_path) 

    file_sgn.Close()




