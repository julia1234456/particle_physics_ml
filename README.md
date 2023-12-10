# Jet Image Classification with Machine Learning

The objective of this project is to use computer vision approach to distinguish between different particle jet images types using Monte Carlo simulations from Pythia8 and ROOT/PyROOT and real data from the Atlas experiment. In particular we apply this framework to distinguish ttbar event (signal) from jj event (noise).

# Repository structure 

particle_physics_ml/
├──ml_jet
    ├── main.py                 
    ├── CONSTANTS.py             
    ├── run_pythia.py            # Pythia8 event generation and jet image extraction via PyROOT
    ├── visualisation.py                  # jet image visualisation (raw, normalized, averaged, variance)
    ├── train_and_inference.py    # prepocessing, model training and inference
    ├── analyze_model.py         # model performance analysis
    └── utils.py                 

# Dependencies 

**Pythia8**  #Monte-Carlo event generator
**ROOT/PyROOT** #CERN's data analysis framework
**NumPy** 
**TensorFlow** 
**Keras**
**Matplotlib**
**scikit-learn**

# Acknowledgment 
The machine learning part is based on the work done by Angelo Monteux. 