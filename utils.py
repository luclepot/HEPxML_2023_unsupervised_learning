import os
import numpy as np
import glob
import pandas as pd
import keras

_LHCO_DATA = None
def load_LHCO():
    """Loads LHCOlympics data into memory. 
    
    Returns:
        dict: in format {dataset_specifier: dataset}
    """
    global _LHCO_DATA
    if _LHCO_DATA is None:
        path = os.path.abspath('data/')
        
        if path is None:
            return None
        
        criteria = 'events_*_features.h5'
        ret = {}
        replace = {'Herwig_qcd_features': 'herwig_qcd', 'Pythia8_v2_Wprime_features': 'wprime', 'Pythia8_v2_qcd_features': 'pythia_qcd'}
        for f in glob.glob(os.path.join(path, criteria)):
            key = replace[f.split('/')[-1].replace('events_anomalydetection_Delphes', '').replace('.h5', '')]
            ret[key] = pd.read_hdf(f)
            add_vars(ret[key])
            
        _LHCO_DATA = ret
        
        _LHCO_DATA['wprime']['signal'] = 1
        _LHCO_DATA['pythia_qcd']['signal'] = 0
        _LHCO_DATA['herwig_qcd']['signal'] = 0

        _LHCO_DATA['wprime']['herwig'] = 0
        _LHCO_DATA['pythia_qcd']['herwig'] = 0
        _LHCO_DATA['herwig_qcd']['herwig'] = 1
        
        _LHCO_DATA = pd.concat(_LHCO_DATA.values()).reset_index(drop=True)
    return _LHCO_DATA

def jet(d, n):
    """get variables for a given jet from a dataset"""
    ret = d[d.columns[d.columns.str.endswith('j' + str(n))]].copy()
    ret.rename({c: c.replace('j' + str(n), '') for c in ret.columns}, axis=1, inplace=True)
    return ret 

def p2(j):
    """calculate magnitude of momentum squared"""
    return j.px**2. + j.py**2. + j.pz**2.

def E(j):
    """calculate energy"""
    return np.sqrt(j.m**2. + p2(j))

def m(j):
    """calculate mass (given energy)"""
    return np.sqrt(j.E**2. - p2(j))

def mjj(d):
    """calculate mjj for a dataset"""
    j1 = jet(d, 1)
    j2 = jet(d, 2)
    j1['E'] = E(j1)
    j2['E'] = E(j2)
    
    return m(j1 + j2)

def pT(d, n):
    """calculate transverse momentum for a dataset"""
    return np.sqrt(d['pxj' + str(n)]**2. + d['pyj' + str(n)]**2.)

def maxmass(d):
    return d.drop([c for c in d if c not in ['mj1', 'mj2']], axis=1).max(axis=1)

def minmass(d):
    return d.drop([c for c in d if c not in ['mj1', 'mj2']], axis=1).min(axis=1)

def tau21(d):
    t21j1 = d.tau2j1/(1e-8 + d.tau1j1)
    t21j2 = d.tau2j2/(1e-8 + d.tau1j2)
    taus = np.asarray([t21j1.values, t21j2.values]).T
    ids = np.argmax(d.drop([c for c in d if c not in ['mj1', 'mj2']], axis=1).values, axis=1)
    idx = keras.utils.to_categorical(ids).astype(bool)
    
    tau21a = taus[idx]
    tau21b = taus[~idx]
    return tau21a, tau21b

def add_vars(d):
    d['mjj'] = mjj(d)
    d['pTj1'] = pT(d, 1)
    d['pTj2'] = pT(d, 2)
    d['maxmass'] = maxmass(d)
    d['minmass'] = minmass(d)
    tau21a, tau21b = tau21(d)
    d['tau21a'] = tau21a
    d['tau21b'] = tau21b