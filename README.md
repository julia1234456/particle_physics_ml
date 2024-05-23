## Jet Image Classification with Machine Learning

We use a computer vision approach to distinguish between different particle jet images types using Monte Carlo simulations from Pythia8 and ROOT/PyROOT and real data from the Atlas experiment. In particular we apply this framework to distinguish ttbar event (signal) from jj event (noise).

## BDT Response and Significance

Looking for rare signals over Standard Model backgrounds (bbjj, zz, ww), we use a Boosted Decision Tree (BDT), a multivariate classifier to distinguish signal from background by combining multiple kinematic variable into a single score (the BDT response). A score close to -1 indicates closeness to background signals, a score close to 1 indicates closeness to rare signal. The objective is to identify the optimal BDT threshold, that maximizes the statistical significance, indicating how confident the model is that the signal is unlikely produced by background fluctuations.

## HistCompare
`ControlPlots` file produces normalized comparison plots for all kinematic and BDT input variables. It overlays the signal and background distributions after applying a BDT pre-selection cut. It is used to visually assess the discriminating power of each variable and validate the analysis selection. \

`StoBRatio` file produces signal-to-background ratio plots for a set of kinematic variables (leading jet mass, pTp_T
pT​, and missing transverse energy). For each background sample, it draws the normalized signal and background distributions with their bin-by-bin ratio. These plots are used to quantify how much the signal and background shapes differ variable by variable. The objective is to identify the most discriminating inputs for the analysis.



## Repository structure 

```
├──ml_ttbar
    ├─ main.py                 
    ├─ constants.py             
    ├─ run_pythia.py  # Pythia8 event generation and jet image extraction via PyROOT
    ├─ visualisation.py # jet image visualisation (raw, normalized, averaged, variance)
    ├─ train_and_inference.py # prepocessing, model training and inference
    ├─ analyze_model.py # model performance analysis
    └─ utils.py 
├──HistCompare
    ├─ AtlasStyle 
        ├─ AtlasLabels.C 
        ├─ AtlasStyle.C
        ├─ AtlasUtils.C
    ├─ ControlPlots.py
    ├─ StoBRatio.py          
├──Significance
    ├─ BDTSign.py



```        

## Dependencies 

- `Pythia8`  (*Monte-Carlo event generator*)
- `ROOT/PyROOT` (*CERN's data analysis framework*)
- `NumPy` 
- `TensorFlow` 
- `Keras`
- `Matplotlib`
- `scikit-learn`

## Acknowledgment 
The machine learning part is based on the work done by A. Monteux. 