import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import illustris_python as il

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, iqr

mpl.rcParams['text.usetex']        = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

fs_og = 20
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 2.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 7.5
mpl.rcParams['ytick.major.size'] = 7.5
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

BLUE = './blue_FMR/'

STARS_OR_GAS = "gas".upper() # stars or gas

whichSim2Tex = {
    'TNG'     :r'${\rm TNG}$',
    'ORIGINAL':r'${\rm Illustris}$',
    'EAGLE'   :r'${\rm EAGLE}$'
}

run, base, out_dir, snapshots = None, None, None, []
snap2z = {}
color  = {}

def switch_sim(WHICH_SIM):
    BLUE_DIR = BLUE + WHICH_SIM + "/"
    if (WHICH_SIM.upper() == "TNG"):
        # TNG
        run       = 'L75n1820TNG'
        base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
        out_dir   = base 
        snapshots = [99,50,33,25,21,17,13,11,8] # 6,4
        snap2z = {
            99:'z=0',
            50:'z=1',
            33:'z=2',
            25:'z=3',
            21:'z=4',
            17:'z=5',
            13:'z=6',
            11:'z=7',
            8 :'z=8',
            6 :'z=9',
            4 :'z=10',
        }
    elif (WHICH_SIM.upper() == "ORIGINAL"):
        # Illustris
        run       = 'L75n1820FP'
        base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
        out_dir   = base
        snapshots = [135,86,68,60,54,49,45,41,38] # 35,32
        snap2z = {
            135:'z=0',
            86 :'z=1',
            68 :'z=2',
            60 :'z=3',
            54 :'z=4',
            49 :'z=5',
            45 :'z=6',
            41 :'z=7',
            38 :'z=8',
            35 :'z=9',
            32 :'z=10',
        }
    elif (WHICH_SIM.upper() == "EAGLE"):
        EAGLE_DATA = BLUE_DIR + 'data/'
        snapshots = [28,19,15,12,10,8,6,5,4] # 3,2
        snap2z = {
            28:'z=0',
            19:'z=1',
            15:'z=2',
            12:'z=3',
            10:'z=4',
             8:'z=5',
             6:'z=6',
             5:'z=7',
             4:'z=8',
             3:'z=9',
             2:'z=10'
        }
    return snapshots, snap2z, BLUE_DIR
        

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02

m_star_min = 8.0
m_star_max = 10.5
m_gas_min  = 8.5

def do(ax_MZR,ax_FMR,ax_sSFRZR,sim,label_size,cmap):
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    z0_alpha = 0.0
    
    snap = snapshots[0] # only interested in z=0

    currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    THRESHOLD = -5.00E-01
    sfms_idx = sfmscut(star_mass, SFR)

    desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                    (star_mass < 1.00E+01**(m_star_max)) &
                    (gas_mass  > 1.00E+01**(m_gas_min))  &
                    (sfms_idx))

    gas_mass  = gas_mass [desired_mask]
    star_mass = star_mass[desired_mask]
    SFR       = SFR      [desired_mask]
    Zstar     = Zstar    [desired_mask]
    Zgas      = Zgas     [desired_mask]
    R_gas     = R_gas    [desired_mask]
    R_star    = R_star   [desired_mask]

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass

    sSFR[~(sSFR > 0.0)] = 1e-16

    star_mass = star_mass[nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    gas_mass      = np.log10(gas_mass)
    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    if snap2z[snap] == 'z=0':
        alphas = np.linspace(0,1,100)
        disp   = np.zeros( len(alphas) )
        a_s    = np.zeros( len(alphas) )
        b_s    = np.zeros( len(alphas) )

        for index, alpha in enumerate(alphas):

            muCurrent = star_mass - alpha*np.log10(SFR) 

            popt = np.polyfit(muCurrent, Z_use, 1)

            a_s[index], b_s[index] = popt

            interp = np.polyval( popt, muCurrent )

            disp[index] = np.std( np.abs(Z_use) - np.abs(interp) )

        argmin = np.argmin(disp)

        z0_alpha = round( alphas[argmin], 2 )

    mu = star_mass - z0_alpha * np.log10(SFR)
    
    ax_MZR.hist2d( star_mass, Z_use, bins=(100,100), norm=LogNorm(), cmap=cmap, rasterized=True )    
    
    ax_FMR.hist2d( mu, Z_use, bins=(100,100), norm=LogNorm(), cmap=cmap, rasterized=True )
    
    ax_sSFRZR.hist2d( star_mass - 1.0*np.log10(SFR), Z_use, bins=(100,100), norm=LogNorm(), cmap=cmap, rasterized=True )
    
    xmin = min( ax_MZR.get_xlim()[0],ax_FMR.get_xlim()[0],ax_sSFRZR.get_xlim()[0]  )
    xmax = max( ax_MZR.get_xlim()[1],ax_FMR.get_xlim()[1],ax_sSFRZR.get_xlim()[1]  )
    
    ymin = min( ax_MZR.get_ylim()[0],ax_FMR.get_ylim()[0],ax_sSFRZR.get_ylim()[0]  )
    ymax = max( ax_MZR.get_ylim()[1],ax_FMR.get_ylim()[1],ax_sSFRZR.get_ylim()[1]  )
    
    ymin = 7.1
    
    if (sim == 'EAGLE'):
        ymin, ymax = 7.9, 9.71
    if (sim == 'TNG'):
        ymin = 7.4
    
    ax_MZR.set_xlim( xmin, xmax )
    ax_FMR.set_xlim( xmin, xmax )
    ax_sSFRZR.set_xlim( xmin, xmax )
    
    ax_MZR.set_ylim( ymin, ymax )
    ax_FMR.set_ylim( ymin, ymax )
    ax_sSFRZR.set_ylim( ymin, ymax )
    
    ax_MZR.set_xlabel(r'$\mu_{0.0} = \log M_* - 0.0 \log{\rm SFR} $',fontsize=label_size)
    ax_FMR.set_xlabel(r'$\mu_{%s}  = \log M_*  - %s \log {\rm SFR}$' %(z0_alpha,z0_alpha),fontsize=label_size)
    ax_sSFRZR.set_xlabel(r'$\mu_{1.0} = \log  M_* - 1.0 \log {\rm SFR}$',fontsize=label_size)
    
    return z0_alpha
    
def line(data, a, b):
    return a*data + b

def sfmscut(m0, sfr0):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 2.0000E-01
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    # Alex added this as a quick bug fix, no idea if it's ``correct''
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool


fig,axs = plt.subplots(3,3,figsize=(13,13))

sims   = ['ORIGINAL','TNG','EAGLE']
cmaps  = [plt.cm.Oranges,plt.cm.Greens,plt.cm.Blues]
cols   = ['C1','C2','C0']

label_size = fs_og * 0.9

alphas = [0, None, 1, 
          0, None, 1, 
          0, None, 1 ]

for index, sim in enumerate(sims):
    print(sim)
    ax = axs[index,:]
    alphas[int(index+1 + 2*index)] = do(ax[0],ax[1],ax[2], sim,label_size,cmaps[index])

for index, ax in enumerate(axs[:,0]):
    ax.set_ylabel( r'$\log {\rm (O/H)} + 12 ~({\rm dex})$', fontsize=label_size )
    ax.text( 0.05, 0.85, whichSim2Tex[sims[index]], transform=ax.transAxes, 
             fontsize=label_size*1.5, color=cols[index] )
    ax.text( 0.05, 0.75, r'$z=0$', transform=ax.transAxes, fontsize=label_size*1.1)
    
for index, ax in enumerate(axs.flatten()):
    ax.text( 0.5, 0.05, r'$\alpha = %s$' %alphas[index], transform=ax.transAxes, ha='center' )
    
relations = [r'${\rm MZR}$',r'${\rm FMR}$',r'${\rm sSFR}^{-1}-{\rm ZR}$']
for index, ax in enumerate(axs[0,:]):
    # ax.text( 0.5, 1.075, relations[index], transform=ax.transAxes, fontsize=label_size*1.1, ha='center' )
    ax.set_title( relations[index], fontsize=label_size*2 )
    
for ax_ in axs[:,1:]:
    for ax in ax_:
        ax.set_yticklabels([])
    

xmin, xmax = np.inf,-np.inf 
for ax_ in axs:
    for ax in ax_:
        tmp_xmin, tmp_xmax = ax.get_xlim()
        
        if (tmp_xmin) < xmin:
            xmin = tmp_xmin
        if (tmp_xmax) > xmax:
            xmax = tmp_xmax
    
for ax_ in axs:
    for ax in ax_:
        ax.set_xlim( xmin, xmax )


plt.tight_layout()
plt.subplots_adjust( hspace=0.25, wspace=0.0 )

plt.savefig( BLUE + 'JWST/' + 'MZR_FMR_sSFRZR' + '.pdf', bbox_inches='tight' )