'''
Explore how to decompose complex flares using MCMC

Try the Davenport et al. (2014) flare template,
the Pugh et al. (2016) QPP template,
and a combination of both.

Use a few big flares from GJ 1243.
'''

import numpy as np
import matplotlib.pyplot as plt
import aflare
import emcee


def QPP(t, p):
    '''
    Eqn. 4 from Pugh et al. (2016)

    Note the data must have the initial flare exponential removed first,
    i.e. Eqn 1 or 2 from Pugh '16, or the Davenport model.

    Parameters
    ----------
    t: 1d-array
        input array of times for the flare
    p: 1d-array
        the fit parameters defining the QPP
    p = [A, B, tau, Per, phi]

    Returns
    -------
    fluxes
    '''

    flux = p[0] * np.exp(-1. * (t - p[1]) / p[2]) * np.cos(2. * np.pi * t / p[3] + p[4])

    return flux


def D1P(t, p):
    '''
    Model with the addition of the Davenport 1-flare template,
    and the Pugh QPP structure.

    p = [tpeak, fwhm, ampl,   # Davenport
         A, B, tau, Per, phi] # Pugh
    '''

    d1 = aflare.aflare1(t, p[0], p[1], p[2])
    p1 = QPP(t, p[3:])

    return d1 + p1

