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
import pandas as pd


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


def D_lnlike(p, x, y, yerr):
    model = aflare.aflare(x, p)
    return -0.5 * np.sum(((y - model)/yerr)**2)


def Danalysis():
    '''
    First exploration: use MCMC with the Davenport et al. (2014) style flare decomposition.

    '''

    # look @ the GJ 1243 flare data...
    datadir = '/Users/james/Dropbox/research_projects/GJ1243-Flares/data/'
    datafile = 'gj1243_master_slc.dat'
    df = pd.read_table(datadir + datafile, delim_whitespace=True, comment='#', header=None,
                       names=['time','flux','err','dflux'])

    medflux = np.nanmedian(df['dflux'])

    # 5 large complex flares that I picked out by hand in 2014, representing different morphologies
    ftimes = [[916.54, 916.66],
              [581.955, 582.025],
              [549.75, 549.895],
              [542.972, 543.015],
              [555.80, 555.865]]

    # my by-eye estimate of the number of components to fit
    Nfl = [8, 7, 9, 4, 7]

    '''
    Plan:
    0. get data only within timespan
    1. subtract local linear continuum off, put in % units
    2. make seed for N flares
    3. run thru MCMC
    4. draw 100 likely flares over the data

    Can put hard limits on parameters (e.g. fwhm) using a ln-likelihood function that
    blows up if a limit is hit.

    '''

    k = 2 # start with the most complex flare

    x = np.where((df['time'] >= ftimes[k][0]) &
                 (df['time'] <= ftimes[k][1]))[0]

    t_fl = df['time'].values[x]
    f_fl = df['flux'].values[x]
    e_fl = df['err'].values[x]

    e_fl = e_fl / medflux

    # fit a line thru the flare, remove, normalize
    c0 = np.median(f_fl[0:5])
    c1 = np.median(f_fl[-5:])
    t0 = np.median(t_fl[0:5])
    t1 = np.median(t_fl[-5:])
    slope = (c1-c0) / (t1-t0)
    inter = c0 - slope * t0
    fline = t_fl * slope + inter

    f_fl = (f_fl - fline) / medflux

    # test plot
    plt.figure()
    plt.errorbar(t_fl, f_fl, yerr=e_fl)
    plt.show()


    time_0 = np.nanmin(t_fl)
    fwhm_0 = 2. / 60. / 24. # 2 min
    ampl_0 = 0.01 # 1% guess

    p0 = np.tile([time_0, fwhm_0, ampl_0], Nfl[k])
    # for l in range(Nfl[k]):
    #     p0


    nsteps = 1000
    nwalkers = 50
    ndim = len(p0)

    # sampler = emcee.EnsembleSampler(nwalkers, ndim, D_lnlike,
    #                                 args=[t_fl, f_fl, e_fl] )
    #
    # sampler.run_mcmc(p0, nsteps)


if __name__ == "__main__":
    # import sys
    Danalysis()
