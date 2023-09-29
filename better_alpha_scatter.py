import sys
import os
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

fs_og = 32
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*2
mpl.rcParams['ytick.major.width'] = 1.5*2
mpl.rcParams['xtick.minor.width'] = 1.0*2
mpl.rcParams['ytick.minor.width'] = 1.0*2
mpl.rcParams['xtick.major.size']  = 7.5*2
mpl.rcParams['ytick.major.size']  = 7.5*2
mpl.rcParams['xtick.minor.size']  = 3.5*2
mpl.rcParams['ytick.minor.size']  = 3.5*2
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

BLUE = './blue_FMR/'

WHICH_SIM    = "eagle".upper() 
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

def do(sim,full_alpha,ax,color,symbol,GLOBAL_MODE,linestyle):
    
    print('Getting individual alphas: %s' %sim)
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    scatter_global = np.zeros( len(snapshots) )
    scatter_local  = np.zeros( len(snapshots) )
    scatter_MZR    = np.zeros( len(snapshots) )
    
    for snap_index, snap in enumerate(snapshots):        
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

        alphas = np.linspace(0,1,100)
        disp   = np.zeros( len(alphas) )
        a_s    = np.zeros( len(alphas) )
        b_s    = np.zeros( len(alphas) )

        for index, alpha in enumerate(alphas):

            muCurrent = star_mass - alpha*np.log10(SFR) 

            popt   = np.polyfit( muCurrent, Z_use, 1 )
            
            a_s[index], b_s[index] = popt
            
            interp = np.polyval( popt, muCurrent )

            disp[index] = np.std( np.abs(Z_use) - np.abs(interp) )
        
        if not (GLOBAL_MODE) and (snap2z[snap] == 'z=0'):
            full_alpha = alphas[ np.argmin(disp) ]
        
        scatter_local[snap_index]  = np.min(disp)  
        scatter_global[snap_index] = disp[ np.argmin( np.abs(alphas - full_alpha )) ]
        
        scatter_MZR[snap_index]    = disp[ 0 ]
    
    redshifts = np.arange(0,9)
    
    ax.plot( redshifts, scatter_local / scatter_global, color=color,
             marker=symbol, label=whichSim2Tex[sim], alpha=0.75, markersize=15,
             linestyle=linestyle, lw=2 )
    
    return scatter_local, scatter_global, scatter_MZR
    
def get_full_alpha(sim):
    print('Getting all z alpha: %s' %sim)
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

    alphas = np.linspace(0,1,100)

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
        interp = np.polyval( popt, mu_fit )
        
        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round( alphas[argmin], 2 )

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

GLOBAL_MODE = True

fig, axs = plt.subplots(1,3,figsize=(20,7),sharey=True,sharex=True)

sims = ['original','tng','eagle']
col  = ['C1','C2','C0']
mark = ['^','*','o']
linestyles = ['solid','dashdot','--']


all_loc, all_glob, all_MZR = [],[],[]

for index, sim in enumerate(sims):
    color = col[index]
    symb  = mark[index]
    sim   = sim.upper()
    if GLOBAL_MODE:
        full_alpha = get_full_alpha(sim)
    else:
        full_alpha = 0.0
    scatter_loc, scatter_glob, scatter_MZR = do(sim,full_alpha,axs[2],color,
                                                symb,GLOBAL_MODE,
                                                linestyle=linestyles[0])
    all_loc.append( scatter_loc )
    all_glob.append( scatter_glob )
    all_MZR.append( scatter_MZR )
    
axs[0].axhline(1, color='gray', linestyle=':', alpha=0.5)

# if GLOBAL_MODE:
#     plt.ylabel( r'$\sigma_{\rm individual} / \sigma_{\rm global}$' )
# else:
# plt.ylabel( r'$\sigma_{\rm individual} / \sigma_{z=0}$' )
for ax in axs:
    ax.set_xlabel( r'${\rm Redshift}$' )

# axs[0].add_artist(leg)
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.0)
# if GLOBAL_MODE:
#     plt.savefig( BLUE + 'JWST/' + 'all_scatter_global' + '.pdf', bbox_inches='tight' )
# else:
#     plt.savefig( BLUE + 'JWST/' + 'all_scatter_z=0' + '.pdf', bbox_inches='tight' )
    
# plt.clf()

redshifts = np.arange(0,9)

# fig = plt.figure(figsize=(10,7))
# ax  = plt.gca()

loc  = np.array( all_loc )
glob = np.array( all_glob )
MZR  = np.array( all_MZR )

ratios1 = loc  / MZR
ratios2 = glob / MZR

illustris = ratios1[0,:]
tng       = ratios1[1,:]
eagle     = ratios1[2,:]

axs[0].plot( redshifts, illustris, color=col[0],
          marker=mark[0], label=whichSim2Tex['ORIGINAL'], 
          alpha=0.75, markersize=15, linestyle=linestyles[1], lw=2
)
axs[0].plot( redshifts, tng      , color=col[1],
          marker=mark[1], label=whichSim2Tex['TNG'], 
          alpha=0.75, markersize=15, linestyle=linestyles[1], lw=2
)
axs[0].plot( redshifts, eagle    , color=col[2],
          marker=mark[2], label=whichSim2Tex['EAGLE'], 
          alpha=0.75, markersize=15, linestyle=linestyles[1], lw=2
)
    
illustris = ratios2[0,:]
tng       = ratios2[1,:]
eagle     = ratios2[2,:]

axs[1].plot( redshifts, illustris, color=col[0],
          marker=mark[0], label=whichSim2Tex['ORIGINAL'], 
          alpha=0.75, markersize=15, linestyle=linestyles[2], lw=2
)
axs[1].plot( redshifts, tng      , color=col[1],
          marker=mark[1], label=whichSim2Tex['TNG'], 
          alpha=0.75, markersize=15, linestyle=linestyles[2], lw=2
)
axs[1].plot( redshifts, eagle    , color=col[2],
          marker=mark[2], label=whichSim2Tex['EAGLE'], 
          alpha=0.75, markersize=15, linestyle=linestyles[2], lw=2
)
    
leg = axs[2].legend( frameon=False,handletextpad=0.25, handlelength=0,
                     labelspacing=0.05, loc='upper right' )
for index, text in enumerate(leg.get_texts()):
    text.set_color(col[index])

# if GLOBAL_MODE:
#     plt.ylabel( r'$\sigma_{\rm global} / \sigma_{\rm MZR}$' )
#     leg_loc = 'upper left'
# else:
#     plt.ylabel( r'$\sigma_{z=0} / \sigma_{\rm MZR}$' )
#     leg_loc = 'upper right'
# plt.xlabel( r'${\rm Redshift}$' )


xmin, xmax = axs[1].get_xlim()
ymin, ymax = axs[1].get_ylim()

axs[0].set_ylabel( r'${\rm Ratio}$' )

axs[2].text( 0.05, 0.85, r'$\sigma_{\rm local} / \sigma_{\rm global}$',
            transform=axs[2].transAxes)
axs[0].text( 0.05, 0.85,r'$\sigma_{\rm local} / \sigma_{\rm MZR}$',
            transform=axs[0].transAxes)
axs[1].text( 0.05, 0.85,r'$\sigma_{\rm global} / \sigma_{\rm MZR}$',
            transform=axs[1].transAxes)

# l1 = plt.plot( redshifts, np.ones(len(redshifts))*10000, color='k', linestyle=linestyles[0],
#           label = r'$\sigma_{\rm local} / \sigma_{\rm global}$' )
# l2 = plt.plot( redshifts, np.ones(len(redshifts))*10000, color='k', linestyle=linestyles[1],
#           label = r'$\sigma_{\rm global} / \sigma_{\rm MZR}$' )

# plot_lines = [l1, l2]
# axs[0].set_xlim(xmin, xmax)
# axs[1].set_xlim(xmin, xmax)
# axs[0].set_ylim(ymin, ymax)

# legend1 = axs[1].legend([l[0] for l in plot_lines], 
#            [r'$\sigma_{\rm local} / \sigma_{\rm global}$',
#             r'$\sigma_{\rm global} / \sigma_{\rm MZR}$'],
#            loc='upper left',
#            frameon=False,handletextpad=0.25, labelspacing=0.05,)
# axs[1].add_artist(legend1)

# leg = plt.legend( frameon=False,handletextpad=0.25, handlelength=0, labelspacing=0.05, loc=leg_loc )
# for index, text in enumerate(leg.get_texts()):
#     text.set_color(col[index])

axs[1].axhline(1, color='gray', linestyle=':', alpha=0.5)

xmin, xmax = axs[1].get_xlim()
ymin, ymax = axs[1].get_ylim()

error_bar = 0.05 # Nominal 1% error bars

axs[0].fill_between( np.arange(-1,11), (1 + error_bar), (1 - error_bar), color='gray', alpha=0.5 )
axs[1].fill_between( np.arange(-1,11), (1 + error_bar), (1 - error_bar), color='gray', alpha=0.5 )
axs[2].fill_between( np.arange(-1,11), (1 + error_bar), (1 - error_bar), color='gray', alpha=0.5 )

axs[0].set_xlim(xmin, xmax)
axs[1].set_xlim(xmin, xmax)
axs[0].set_ylim(ymin, ymax)

axs[0].set_xticks([0,2,4,6,8])

plt.tight_layout()
plt.subplots_adjust(wspace=0.0)
# if GLOBAL_MODE:
#     plt.savefig( BLUE + 'JWST/' + 'MZR_scatter_global' + '.pdf', bbox_inches='tight' )
# else:
#     plt.savefig( BLUE + 'JWST/' + 'MZR_scatter_z=0' + '.pdf', bbox_inches='tight' )

plt.savefig( BLUE + 'JWST/' + 'all_scatters' + '.pdf', bbox_inches='tight' )