
import os 
import fastjet 
import sys
pythia_path = 'pythia8309/'
cfg = open(pythia_path+"/Makefile.inc")
cfg = os.path.join(pythia_path, 'Makefile.inc')
lib=pythia_path+'lib'
lib = os.path.join(pythia_path, 'lib')


for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
    sys.path.insert(0, lib)
sys.path.append(lib)
os.environ["PYTHONPATH"] = f"{lib}:{os.environ.get('PYTHONPATH', '')}"
os.system("source ~/.bashrc")

import pythia8
import numpy as np
from CONSTANTS import *
from utils import *


def write_mg_cards(PTRANGE, delta=10, nevents=200000, extra=''):
    """
    Write two Madgraph cards for production of ttbar and jj events.
    Apply generator-level cut on the particle momenta given by PTRANGE,
    allowing for a tolerance delta.
    If argument extra is specified, append it to output directories
    """
    with open('generate_tt.mg5', 'w') as f:
        f.write(
"""
    generate p p > t t~
    output jets_tt{3}
    launch
    madspin=none
    done
    set nevents {2}
    set pt_min_pdg {{ 6: {0} }}
    set pt_max_pdg {{ 6: {1} }}
    decay t > w+ b, w+ > j j
    decay t~ > w- b~, w- > j j
    done
""".format(PTRANGE[0] - delta, PTRANGE[1] + delta, nevents, extra)
        )

    with open('generate_qcd.mg5','w') as f:
        f.write("""
    generate p p > j j
    output jets_qcd{3}
    launch
    done
    set nevents {2}
    set ptj {0}
    set ptjmax {1}
    done    
    """.format(PTRANGE[0]-delta, PTRANGE[1]+delta, nevents, extra))

def main_write_mg():
    madgraphdir = 'MG5_aMC_v2_9_15/'
    assert os.path.isdir(madgraphdir)

    cwd = os.getcwd()

    ### write madgraph cards, run madgraph   -- PTRANGE=500-700
    write_mg_cards([250, 300], nevents=nevents)

    ret = os.system('cd %s; bin/mg5_aMC  %s' % (madgraphdir, os.path.join(cwd, 'generate_tt.mg5' )))
    assert ret == 0

    ret = os.system('cd %s; bin/mg5_aMC  %s' % (madgraphdir, os.path.join(cwd, 'generate_qcd.mg5')))
    assert ret == 0

def make_image_jet(start_x, stop_x, start_y, stop_y, jet_constituents):
    etas = set(jet_constituents[:, 1])
    phis = set(jet_constituents[:, 2])

    if len(etas) < 2 or len(phis) < 2:
        return None, (None, None)

    optimal_w = MAX_WIDTH
    optimal_h = MAX_HEIGHT

    xedges = np.linspace(start_x, stop_x, optimal_w + 1)
    yedges = np.linspace(start_y, stop_y, optimal_h + 1)

    #histo, xedges, yedges = np.histogram2d(jet_constituents[:,1], jet_constituents[:,2], bins=(xedges,yedges), weights=jet_constituents[:,0])
    #return np.flipud(histo.T), (xedges, yedges)

    # The following gives results slightly smoother than np.histogram2d()
    histo = consts_to_image(jet_constituents, xedges, yedges)
    return histo, (xedges, yedges)

def make_image_leading_jet(leading_jet_constituents, subjets):
    """
    Jet and constituents are passed as pythia vec4 objects.
    Restricts image grid to within a variable range around jet center.
    Returns pT-weighted histogram, and tuple with histogram grid.
    """
    # Extract phi and eta coordinates of the leading jet
    jet_phi = subjets[0].phi()
    jet_eta = subjets[0].eta()


    jet_constituents = np.array([[c.pt(), c.eta() - jet_eta, mod_pi(c.phi() - jet_phi)] for c in leading_jet_constituents])

    return make_image_jet(
        eta_range_start,
        eta_range_end,

        phi_range_start,
        phi_range_end,

        jet_constituents
    )


def run_pythia_get_images(lhe_file_name, PTRANGE=[250, 300], PTRANGE2=None, nevents=10**6):
    """
    Take an LHE file, run pythia on it, outputs images.
    For each event, cluster jets, check if the two highest pT jets are in PTRANGE and PTRANGE2,
    and make 2D histograms of the leading jet and of the whole event.
    """
    # unzip LHE file if it is zipped
    if lhe_file_name.endswith('gz') and not os.path.isfile(lhe_file_name.split('.gz')[0]):
        os.system('gunzip < {} > {}'.format(lhe_file_name, lhe_file_name.split('.gz')[0]))

    lhe_file_name = lhe_file_name.split('.gz')[0]

    if not os.path.isfile(lhe_file_name):
        raise Exception('no LHE file')

    if PTRANGE2 is None:
        PTRANGE2 = PTRANGE


    

    pythia = pythia8.Pythia()
    ### read LHE input file
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = " + lhe_file_name)

    pythia.init()
    #--------------------
    """
    # Initialize Delphes
    os.system('./MG5_aMC_v2_9_15/Delphes/DelphesSTDHEP delphes_card_ATLAS.tcl delphes_output_ATLAS.root')  # Run Delphes

    # Access Delphes output
    delphes_file_name = 'delphes_output_ATLAS.root'
    if not os.path.isfile(delphes_file_name):
        raise Exception('Delphes output not found')

    #--------------------
    
    """

    # outputs: lists of leading jet images
    leading_jet_images = []

    ### Begin event loop. Generate event. Skip if error or file ended. Print counter
    for iEvent in range(nevents):
        if not pythia.next():
            continue

        pythia_event = pythia.event

        if iEvent % 10 == 0:
            print(iEvent)

        ### Cluster jets. List first few jets. Excludes neutrinos by default
        jet_ptcl = []

        for i_ptcl in range(pythia_event.size()):
            pythia_ptcl = pythia_event[i_ptcl]
            if not pythia_ptcl.isFinal():
                continue

            # Skip neutrinos, PDGid = 12, 14, 16
            pdgid = abs(pythia_ptcl.id())
            if (pdgid == 12 or pdgid == 14 or pdgid == 16):
                continue

            fastjet_ptcl = fastjet.PseudoJet(pythia_ptcl.px(), pythia_ptcl.py(), pythia_ptcl.pz(), pythia_ptcl.e())
            # assert fastjet_ptcl.px() == pythia_ptcl.px()
            # assert fastjet_ptcl.py() == pythia_ptcl.py()
            # assert fastjet_ptcl.pz() == pythia_ptcl.pz()
            # assert fastjet_ptcl.e()  == pythia_ptcl.e()
            jet_ptcl.append(fastjet_ptcl)

        # Cluster jets using anti_kt algorithm with R = 1.0
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)

        # Filter that decomposes a jet into subjets using kt algorithm with R = 0.3, and then keeps only the subjets that carry >= 5% pT of the jet.
        trimmer = fastjet.Filter(
            fastjet.JetDefinition(fastjet.kt_algorithm, 0.3),
            fastjet.SelectorPtFractionMin(0.05)
        )
 
        cs = fastjet.ClusterSequence(jet_ptcl, jet_def)

        # Get all jets with pt >= 10.0, sorted by pT (descending)
        jets = fastjet.sorted_by_pt(cs.inclusive_jets(10.0))
        jets = [trimmer(jet) for jet in jets]

        if len(jets) < 2:
            continue

        if not (PTRANGE[0] < jets[0].pt() < PTRANGE[1] and PTRANGE2[0] < jets[1].pt() < PTRANGE2[1]):
            continue

        leading_jet = jets[0]
        if not leading_jet.constituents():
            continue

        constituents = sorted(leading_jet.constituents(), key=lambda const: const.pt(), reverse=True)

        subjets = sorted(leading_jet.pieces(), key=lambda subjet: subjet.pt(), reverse=True)
        assert subjets
        

        hh, _ = make_image_leading_jet(constituents, subjets)
        if hh is None:
            continue

        leading_jet_images.append(hh)

   

    return np.array(leading_jet_images, dtype=object)


def main_run_pythia():
    madgraphdir = 'MG5_aMC_v2_9_15/'
    assert os.path.isdir(madgraphdir)

    outdir = 'images_out/' 
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    ### run pythia
    lhe_file_name = madgraphdir + 'jets_tt/Events/run_01_decayed_1/unweighted_events.lhe.gz'
    leading_jet_images = run_pythia_get_images(lhe_file_name, PTRANGE=[250, 300], PTRANGE2=[200, 300], nevents=nevents)

    np.savez(outdir + 'tt_leading_jet.npz', leading_jet_images)
 
    del leading_jet_images

    lhe_file_name = madgraphdir + 'jets_qcd/Events/run_01/unweighted_events.lhe.gz'
    leading_jet_images = run_pythia_get_images(lhe_file_name, PTRANGE=[250, 300], PTRANGE2=[200, 300], nevents=nevents)

    np.savez(outdir + 'qcd_leading_jet.npz', leading_jet_images)
    
