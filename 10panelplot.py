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

    
def do(sim,LFMR,all_z_fit):
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
    
    for snap in snapshots:
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

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
    
    min_alpha = round( alphas[argmin], 2 )
    best_mu   = star_mass - min_alpha * np.log10(SFR)
    best_line = a_s[argmin] * best_mu + b_s[argmin]
    
    if LFMR:
        min_alpha, _a_, _b_ = get_z0_alpha(sim)
        best_mu = star_mass - min_alpha*np.log10(SFR)
        if (all_z_fit):
            _a_, _b_ = np.polyfit(best_mu, Z_use, 1)
        best_line = _a_ * best_mu + _b_
    
    WHICH_SIM_TEX = {
        "TNG":r"${\rm TNG}$",
        "ORIGINAL":r"${\rm Illustris}$",
        "EAGLE":r"${\rm EAGLE}$"
    }

    unique, n_gal = np.unique(redshifts, return_counts=True)

    newcolors   = plt.cm.rainbow(np.linspace(0, 1, len(unique)-1))
    CMAP_TO_USE = ListedColormap(newcolors)

    plt.clf()
    mpl.rcParams['font.size'] = 50
    mpl.rcParams['xtick.major.width'] = 1.5 *2.5
    mpl.rcParams['ytick.major.width'] = 1.5 *2.5
    mpl.rcParams['xtick.minor.width'] = 1.0 *2.5
    mpl.rcParams['ytick.minor.width'] = 1.0 *2.5
    mpl.rcParams['xtick.major.size']  = 7.5 *2.5
    mpl.rcParams['ytick.major.size']  = 7.5 *2.5
    mpl.rcParams['xtick.minor.size']  = 3.5 *2.5
    mpl.rcParams['ytick.minor.size']  = 3.5 *2.5
    mpl.rcParams['axes.linewidth']    = 2.25*2.5
    fig = plt.figure(figsize=(30,20))

    gs  = gridspec.GridSpec(4, 7, width_ratios = [0.66,0.66,0.66,0.35,1,1,1],
                             height_ratios = [1,1,1,0.4], wspace = 0.0, hspace=0.0)

    axBig = fig.add_subplot( gs[:,:3] )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,weights=redshifts,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    mappable = axBig.pcolormesh( xedges, yedges, hist, vmin = 0, vmax = 8, cmap=CMAP_TO_USE, rasterized=True )

    cbar = plt.colorbar( mappable, label=r"${\rm Redshift}$", orientation='horizontal' )
    cbar.ax.set_yticklabels(np.arange(0,9))

    axBig.plot( best_mu, best_line, color='k', lw=6.0 )

    if (STARS_OR_GAS == "GAS"):
        plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
    elif (STARS_OR_GAS == "STARS"):
        plt.ylabel(r'$\log(Z_* [Z_\odot])$')
    plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

    if (LFMR):
        axBig.text( 0.75, 0.15, r"${\rm Local~ FMR}$", transform=axBig.transAxes, ha='center' )
    else:
        axBig.text( 0.75, 0.15, r"${\rm Global~ FMR}$", transform=axBig.transAxes, ha='center' )
    
    if all_z_fit:
        axBig.text( 0.75, 0.1, r"${\rm All~}z~{\rm fit}$", transform=axBig.transAxes, ha='center' )
    else:
        axBig.text( 0.75, 0.1, r"$z=0~{\rm Calibrated}$", transform=axBig.transAxes, ha='center' )
    axBig.text( 0.05, 0.9, "%s" %WHICH_SIM_TEX[sim], transform=axBig.transAxes )

    # plt.tight_layout()
    # plt.savefig( 'test_all_z' )

    Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,bins=(100,100))
    Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])

    percentage = 0.01
    xmin, xmax = np.min(best_mu)*(1-percentage), np.max(best_mu)*(1+percentage)
    ymin, ymax = np.min(Z_use)  *(1-percentage), np.max(Z_use)  *(1+percentage)
    
    axBig.set_xlim(xmin,xmax)
    axBig.set_ylim(ymin,ymax)
    
    Hist1 = np.transpose(Hist1)
    Hist2 = np.transpose(Hist2)

    hist = Hist1/Hist2

    ax_x = 0
    ax_y = 4

    axInvis = fig.add_subplot( gs[:,3] )
    axInvis.set_visible(False)

    small_axs = []
    ylab_flag = True

    for index, time in enumerate(unique):
        ax = fig.add_subplot( gs[ax_x, ax_y] )

        small_axs.append(ax)

        if (ylab_flag):
            if (STARS_OR_GAS == "GAS"):
                ax.set_ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$',fontsize=36 )
            elif (STARS_OR_GAS == "STARS"):
                ax.set_ylabel(r'$\log(Z_* [Z_\odot])$',fontsize=36 )

            ylab_flag = False

        if (ax_x == 2):
            ax.set_xlabel( r'$\mu_{%s}$' %(min_alpha),
                           fontsize=36 )

        if (ax_y == 5 or ax_y == 6):
            ax.set_yticklabels([])
        if (ax_y == 0 or ax_y == 1):
            ax.set_xticklabels([])

        ax_y += 1
        if (ax_y == 7):
            ax_y = 4
            ax_x += 1
            ylab_flag = True

        mask = (redshifts == time)

        ax.pcolormesh( xedges, yedges, hist, alpha=0.5, vmin = 0, vmax = 1.5, cmap=plt.cm.Greys, rasterized=True )

        current_mu    =   best_mu[mask]
        current_Z     =     Z_use[mask]
        current_smass = star_mass[mask]
        current_gmass =  gas_mass[mask]
        current_SFR   =       SFR[mask]
        current_Rgas  =     R_gas[mask]
        current_Rstar =    R_star[mask]

        Hist1, current_x, current_y = np.histogram2d(current_mu,current_Z,bins=(100,100))
        Hist2, _        , _         = np.histogram2d(current_mu,current_Z,bins=[current_x,current_y])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        current_hist = Hist1/Hist2

        vmin = 1 - time
        vmax = 9 - time

        ax.pcolormesh( current_x, current_y, current_hist, vmin = vmin, vmax = vmax, cmap=CMAP_TO_USE, rasterized=True )

        ax.plot( best_mu, best_line, color='k', lw=4.5 )

        ax.text( 0.65, 0.1, r"$z = %s$" %int(time), transform=plt.gca().transAxes )

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    plt.tight_layout()
    if LFMR:
        if all_z_fit:
            plt.savefig( BLUE + 'JWST/' + '%s_big_fig_LFMR_all_z.pdf' %sim, bbox_inches='tight' )
        else:
            plt.savefig( BLUE + 'JWST/' + '%s_big_fig_LFMR.pdf' %sim, bbox_inches='tight' )
    else:
        plt.savefig( BLUE + 'JWST/' + '%s_big_fig.pdf' %sim, bbox_inches='tight' )
    plt.clf()
        
def get_z0_alpha(sim):
    print('Getting z=0 alpha: %s' %sim)
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    snap = snapshots[0]

    currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

    Zgas      = np.load( currentDir + 'Zgas.npy' )
    Zstar     = np.load( currentDir + 'Zstar.npy' ) 
    star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
    gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
    SFR       = np.load( currentDir + 'SFR.npy' )
    R_gas     = np.load( currentDir + 'R_gas.npy' )
    R_star    = np.load( currentDir + 'R_star.npy' )

    sfms_idx = sfmscut(star_mass, SFR)

    desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                    (star_mass < 1.00E+01**(m_star_max)) &
                    (gas_mass  > 1.00E+01**(m_gas_min))  &
                    (sfms_idx))

    gas_mass  =  gas_mass[desired_mask]
    star_mass = star_mass[desired_mask]
    SFR       =       SFR[desired_mask]
    Zstar     =     Zstar[desired_mask]
    Zgas      =      Zgas[desired_mask]
    R_gas     =     R_gas[desired_mask]
    R_star    =    R_star[desired_mask]

    all_Zgas     += list(Zgas     )
    all_Zstar    += list(Zstar    )
    all_star_mass+= list(star_mass)
    all_gas_mass += list(gas_mass )
    all_SFR      += list(SFR      )
    all_R_gas    += list(R_gas    )
    all_R_star   += list(R_star   )
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)
    a_s    = np.zeros( len(alphas) ) # y = ax + b
    b_s    = np.zeros( len(alphas) )

    disps = np.ones(len(alphas)) * np.nan
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        muCurrent  = star_mass - alpha*np.log10( SFR )

        mu_fit = muCurrent
        Z_fit  = Z_use
        
        popt = np.polyfit(mu_fit, Z_fit, 1)
        a_s[index], b_s[index] = popt
        interp = np.polyval( popt, mu_fit )
        
        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round( alphas[argmin], 2 ), a_s[argmin], b_s[argmin]
    
def line(data, a, b):
    return a*data + b

def fourth_order( x, a, b, c, d, e ):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def third_order( x, a, b, c, d ):
    return a + b*x + c*x**2 + d*x**3

def scatter_at_fixed_mu( mu, Z ):
    
    start = np.min(mu)
    end   = np.max(mu)
    
    width = 0.3
    
    current = start
    
    scatter = []
    
    while (current < end):
        
        mask = ( ( mu > current ) &
                 ( mu < current + width) )
        
        scatter.append( np.std( Z[mask] ) * len(Z[mask]) )
        
        current += width
        
    return np.array(scatter)

def sfmscut(m0, sfr0, THRESHOLD=-5.00E-01):
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
        idxb  = (ssfrb - rdg) > THRESHOLD
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
        idxb  = (ssfrb - ssfrlin[i]) > THRESHOLD
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool        

sims = ['original','tng','eagle']

LFMR = True
all_z_fit = True

if not LFMR:
    all_z_fit = False

for index, sim in enumerate(sims):
    sim = sim.upper()
    print(sim)
    do(sim, LFMR, all_z_fit)