#!/usr/bin/env python

import sys, os
if not 'PYTHIA8DATA' in os.environ:
    #os.environ['PYTHIA8DATA'] = '/home/hep/tipton/cos26/local/src/pythia8230/share/Pythia8/xmldoc'
    os.environ['PYTHIA8DATA'] = '/home/cshimmin/analysis/pythia/numpythia/numpythia/src/extern/pythia8230/share/Pythia8/xmldoc'

from numpythia import Pythia, hepmc_write
from numpythia.testcmnd import get_cmnd
from numpythia import STATUS, ABS_PDG_ID, HAS_END_VERTEX

from pyjet import cluster

import numpy as np
import argparse

def charge(pdgid):
    absid = abs(pdgid)

    charged = (
            11,13,15, #leptons
            24, # Wboson
            2212, # proton
            211,321,
            )
    neutral = (
            12,14,16, # neutrinos
            21,22,23,25, # bosons
            2112,
            111,130
            )

    if absid in charged:
        return pdgid/absid
    elif absid in neutral:
        return 0
    else:
        raise Exception("Unknown pdgid %d"%pdgid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pythia.cmnd', help="The config file to use")
    parser.add_argument('--seed', type=int, default=0, help="The random seed to initialize with")
    parser.add_argument('--nevent', type=int, default=10, help="The number of events to generate")
    parser.add_argument('--out', help="The output npy file")
    args = parser.parse_args()

    try:
        os.makedirs(os.path.dirname(args.out))
    except OSError as e:
        import errno
        if e.errno != errno.EEXIST:
            raise

    pythia = Pythia(args.config, random_state=args.seed)

    # for some stupid reason there has to be at least two terms for
    # this selection thing to work.
    sel = (STATUS==1) & ~HAS_END_VERTEX & (ABS_PDG_ID != 12) & (ABS_PDG_ID != 14) & (ABS_PDG_ID != 16)
    
    dtype = [('ntrk1',int),('ntrk2',int)] + [('wt_%s'%l,float) for l in pythia.weight_labels]
    results = []
    for ievt,event in enumerate(pythia(events=args.nevent)):
        if ievt%100==0:
            print("Processing event %d/%d" % (ievt, args.nevent))

        weights = list(event.weights)

        particles = event.all(sel)
        
        # make jets to filter for pT>500
        sequence = cluster(particles, R=0.4, ep=True, p=-1)
        jets = sequence.inclusive_jets()
        jets.sort(key=lambda j: j.pt, reverse=True)
        j1, j2 = jets[:2]

        tracks1 = list(filter(lambda pj: charge(pj.pdgid)!=0, j1.constituents()))
        tracks2 = list(filter(lambda pj: charge(pj.pdgid)!=0, j2.constituents()))

        results.append(tuple([ len(tracks1), len(tracks2) ] + weights))

    if args.out:
        print("Saving to", args.out)
        print(dtype)
        print(np.array(results))
        print(np.array(results).shape)
        np.save(args.out, np.array(results, dtype=dtype))
