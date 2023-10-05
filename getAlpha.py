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

WHICH_SIM    = "eagle".upper() 
STARS_OR_GAS = "gas".upper() # stars or gas

PLOT        = True
GIF         = False
COMBINE_ALL = True
BEST_FIG    = True
OVERWRITE   = False
UNCERTAIN   = True
DECREMENT   = False
WEIRD_PLOT  = False
FIT_WITH_LOW_REDSHIFT_ONLY = False
DUAL_CRITERIA = False

if (OVERWRITE):
    print('')
    print('#'*40)
    print('Overwritting entire file alpha_%s_values.h5...' %STARS_OR_GAS.lower())
    print('You will need to redo all simulations with OVERWRITE = False')
    alpha_file = h5py.File(BLUE + 'alpha_%s_values.h5' %STARS_OR_GAS.lower(),'w')
    uncertain  = h5py.File(BLUE + 'alpha_uncertainty_%s_values.h5' %STARS_OR_GAS.lower(), 'w')
    if (DECREMENT):
        decrement_file = h5py.File(BLUE + 'alpha_%s_decrement.h5' %STARS_OR_GAS.lower(), 'w')
    print('#'*40)
    print('\n')
else:
    print('')
    print('#'*40)
    print('I am not overwriting the file alpha_%s_values.h5' %STARS_OR_GAS.lower())
    alpha_file = h5py.File(BLUE + 'alpha_%s_values.h5' %STARS_OR_GAS.lower(),'a')
    uncertain  = h5py.File(BLUE + 'alpha_uncertainty_%s_values.h5' %STARS_OR_GAS.lower(),'a')
    if (DECREMENT):
        decrement_file = h5py.File(BLUE + 'alpha_%s_decrement.h5' %STARS_OR_GAS.lower(), 'a')
    print('#'*40)
    print('\n')

BLUE_DIR = BLUE + WHICH_SIM + "/"

run, base, out_dir, snapshots = None, None, None, []
snap2z = {}
color  = {}

if (WHICH_SIM.upper() == "TNG"):
    # TNG
    run       = 'L75n1820TNG'
    base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
    out_dir   = base 
    # snapshots = [99,50,33,25,21,17,13,11,8,6,4]
    snapshots = [99,50,33,25,21,17,13,11,8]
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
    color = {
        99:'blue',
        50:'cyan',
        33:'orange',
        25:'red',
        21:'blue',
        17:'cyan',
        13:'orange',
        11:'red',
        8 :'orange',
        6 :'red',
        4 :'blue'
    }
    

elif (WHICH_SIM.upper() == "TNG-2"):
    # TNG
    run       = 'L75n910TNG'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/' 
    out_dir   = base 
    # snapshots = [99,50,33,25,21,17,13,11,8,6,4]
    snapshots = [99,50,33,25,21,17,13,11,8]
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
    color = {
        99:'blue',
        50:'cyan',
        33:'orange',
        25:'red',
        21:'blue',
        17:'cyan',
        13:'orange',
        11:'red',
        8 :'orange',
        6 :'red',
        4 :'blue'
    }
elif (WHICH_SIM.upper() == "TNG-3"):
    # TNG
    run       = 'L75n455TNG'
    base      = '/orange/paul.torrey/alexgarcia/' + run + '/' 
    out_dir   = base 
    # snapshots = [99,50,33,25,21,17,13,11,8,6,4]
    snapshots = [99,50,33,25,21,17,13,11,8]
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
    color = {
        99:'blue',
        50:'cyan',
        33:'orange',
        25:'red',
        21:'blue',
        17:'cyan',
        13:'orange',
        11:'red',
        8 :'orange',
        6 :'red',
        4 :'blue'
    }
    
elif (WHICH_SIM.upper() == "ORIGINAL"):
    # Illustris
    run       = 'L75n1820FP'
    base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
    out_dir   = base
    # snapshots = [135,86,68,60,54,49,45,41,38,35,32]
    snapshots = [135,86,68,60,54,49,45,41,38]
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
    color = {
        135:'blue',
        86 :'cyan',
        68 :'orange',
        60 :'red',
        54 :'blue',
        49 :'cyan',
        45 :'orange',
        41 :'red',
        38 :'orange',
        35 :'red',
        32 :'blue'
    }
    
elif (WHICH_SIM.upper() == "ORIGINAL-2"):
    # Illustris
    run       = 'L75n910FP'
    base      = '/orange/paul.torrey/alexgarica/' + run + '/'
    out_dir   = base
    # snapshots = [135,86,68,60,54,49,45,41,38,35,32]
    snapshots = [135,86,68,60,54,49,45,41,38]
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
    color = {
        135:'blue',
        86 :'cyan',
        68 :'orange',
        60 :'red',
        54 :'blue',
        49 :'cyan',
        45 :'orange',
        41 :'red',
        38 :'orange',
        35 :'red',
        32 :'blue'
    }
    
elif (WHICH_SIM.upper() == "ORIGINAL-3"):
    # Illustris
    run       = 'L75n455FP'
    base      = '/orange/paul.torrey/alexgarica/' + run + '/'
    out_dir   = base
    # snapshots = [135,86,68,60,54,49,45,41,38,35,32]
    snapshots = [135,86,68,60,54,49,45,41,38]
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
    color = {
        135:'blue',
        86 :'cyan',
        68 :'orange',
        60 :'red',
        54 :'blue',
        49 :'cyan',
        45 :'orange',
        41 :'red',
        38 :'orange',
        35 :'red',
        32 :'blue'
    }
    
elif (WHICH_SIM.upper() == "EAGLE"):
    # snapshots = [28,19,15,12,10,8,6,5,4,3,2]
    snapshots = [28,19,15,12,10,8,6,5,4]
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
    color = {
        28:'blue',
        19:'cyan',
        15:'orange',
        12:'red',
        10:'blue',
         8:'cyan',
         6:'orange',
         5:'red',
         4:'blue',
         3:'cyan',
         2:'orange',
    }

COLORCUT = True
CMAX = -10
CMIN = -9

sSFRcut = {
    'z=0':(-10.5,-9.5),
    'z=1':(-10,-9),
    'z=2':(-9.5,-8.5),
    'z=3':(-9,-8)
}

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

def do():
    try:
        currentSimGroup             = alpha_file.create_group('%s' %WHICH_SIM)
        currentSimGroup_uncertainty = uncertain.create_group( '%s' %WHICH_SIM)
        if (DECREMENT):
            currentSimGroup_decrement = decrement_file.create_group('%s' %WHICH_SIM)
    except:
        currentSimGroup             = alpha_file.get('%s' %WHICH_SIM)
        currentSimGroup_uncertainty = uncertain.get( '%s' %WHICH_SIM)
        if (DECREMENT):
            currentSimGroup_decrement = decrement_file.get('%s' %WHICH_SIM)
    
    tot_n_gal = 0
    n_gal = []
    
    for snap in snapshots:

        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )

        # Nominal threshold = -5.000E-01
        sfms_idx = sfmscut(star_mass, SFR, THRESHOLD=-5.00E-01)

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
        
        tot_n_gal += len( star_mass )
        n_gal.append( len( star_mass ) )
        
        print(tot_n_gal)
        print(len(star_mass))
        
        alphas = np.linspace(0,1,100)
        used_alphas = np.ones(len(alphas)) * np.nan
        
        disps = np.ones(len(alphas)) * np.nan
        
        best_coefs = None
        min_alpha  = -1.0
        min_disp   = np.inf
        best_mu    = None
        best_line  = None
        min_index  = -1
        
        print(STARS_OR_GAS)
        
        if (STARS_OR_GAS == "GAS"):
            Z_use = Zgas
        elif (STARS_OR_GAS == "STARS"):
            Z_use = Zstar
        else:
            break
        
        for index, alpha in enumerate(alphas):
            
            muCurrent = star_mass - alpha*np.log10(SFR) 

            popt = np.polyfit(muCurrent, Z_use, 1)
            
            # popt, pcov = curve_fit(line, muCurrent, Z_use) # did not work for Illustris gas-phase for some reason????
            
            interp = np.polyval( popt, muCurrent )            
            resids = np.std( np.abs(Z_use) - np.abs(interp) )
            
            disps[index] = resids
            used_alphas[index] = alpha

            ####################### MAKE GIF #######################
            if (GIF):
                plt.clf()

                fig, ax = plt.subplots(1,2,figsize=(10,6))

                ax1 = ax[0]
                ax2 = ax[1]

                ax1.hist2d(muCurrent, Z_use, bins=(100,100), norm=LogNorm())
                                
                ax1.scatter(muCurrent, interp, color='k')
                ax1.set_xlim(8.0,11.0)

                ax1.set_xlabel(r'$\mu = \log M_* - %s \log {\rm SFR}$' %round(alpha,2) )
                if (STARS_OR_GAS == "GAS"):
                    ax1.set_ylabel(r'$\log {\rm O/H} + 12$ (dex)')
                elif (STARS_OR_GAS == "STARS"):
                    ax1.set_ylabel(r'$\log Z_*/Z_\odot$')

                ax2.text(0.5,0.5 ,r'{\rm %s}' %(WHICH_SIM),  transform=ax2.transAxes)
                ax2.text(0.5,0.45,'%s' %(snap2zTex[snap]),  transform=ax2.transAxes)
                    
                ax2.plot( used_alphas, disps  )

                ax2.set_xlim(0,1)
#                 ax2.set_ylim(0.1,0.4)
#                 ax2.set_yticklabels([])

                ax2.set_xlabel(r'$\alpha$')
                ax2.set_ylabel(r'${\rm Dispersion}\ {\rm from}\ {\rm Median}$')

                plt.tight_layout()
                plt.savefig('./blue_FMR/alpha_gif/%03d' %index)
            ####################### MAKE GIF #######################   
            
            if (np.nansum(resids) < min_disp):
                min_disp  = np.nansum(resids)
                min_alpha = round(alpha,2)
                best_mu   = muCurrent
                best_line = interp
                min_index = index
                
#                 best_coefs = popt

        print('%s: %s, best alpha_%s = %s' %(WHICH_SIM, snap2z[snap], STARS_OR_GAS.lower(), min_alpha))
        if (GIF):
            break
        try:
            currentSimGroup.create_dataset('snap%s_alpha' %snap, data = min_alpha)
        except:
            print('#'*100)
            print('FILE ALREADY EXISTS')
            print('#'*100)
        if (WEIRD_PLOT):
            plt.clf()
            plt.axvline(min_alpha,color='k',linestyle='--')
            plt.plot(alphas,disps)

            plt.xlabel(r'$\alpha$')
            plt.ylabel('Dispersion from Median')

            plt.tight_layout()

            plt.savefig('alpha_test_1_%s'%snap)

            plt.clf()

            plt.hist2d(star_mass - min_alpha*np.log10(SFR), Z_use, bins=(100,100), norm=LogNorm())

#             plt.scatter(best_xs, best_medians, color='k')

            plt.xlabel(r'$\mu = \log M_* - %s \log {\rm SFR}$' %round(min_alpha,2) )
            if (STARS_OR_GAS == "GAS"):
                plt.ylabel(r'$\log {\rm O/H} + 12$ (dex)')
            elif (STARS_OR_GAS == "STARS"):
                plt.ylabel(r'$\log Z_*/Z_\odot$')

            plt.tight_layout()
            plt.savefig('alpha_test_2_%s'%snap)
        if (BEST_FIG):
            plt.clf()
            fig, axs = plt.subplots(2,1,figsize=(5,8))
            
            axs[0].plot( alphas, disps, lw=2 )
            axs[0].axhline( min_disp, color='r', linestyle='--' )
            axs[0].axvline( min_alpha, color='r',linestyle='--' )
            axs[0].set_xlim(0,1)
            axs[0].set_ylabel(r'${\rm Dispersion}$')
            axs[0].set_xlabel(r'$\alpha$')
            
            
            WHICH_SIM_TEX = {
                "TNG":r"${\rm TNG}$",
                "ORIGINAL":r"${\rm Illustris}$",
                "EAGLE":r"${\rm EAGLE}$"
            }
            
            axs[0].text( 0.4, 0.6 , r'%s' %WHICH_SIM_TEX[WHICH_SIM], transform=axs[0].transAxes )
            axs[0].text( 0.4, 0.5 , r'$%s$' %snap2z[snap], transform=axs[0].transAxes )
            
            axs[1].hist2d( best_mu, Z_use, bins=(100,100), norm=LogNorm() )
            axs[1].plot( best_mu, best_line, color='k', lw=2 )
            axs[1].set_ylabel(r'$\log(Z_* [Z_\odot])$')
            axs[1].set_xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))
            
            axs[1].text( 0.375 , 0.2  , r'${\rm Minimum~dispersion~fit}$', 
                        transform=axs[1].transAxes, fontsize=14 )
            axs[1].text( 0.55  , 0.125, r'$\alpha=%s$' %min_alpha        , 
                        transform=axs[1].transAxes, fontsize=14 )

            if (UNCERTAIN):
                width = min_disp * 1.01
    
                within_uncertainty = alphas[ (disps < width) ]
    
                min_uncertain = within_uncertainty[0]
                max_uncertain = within_uncertainty[-1]
            
                print "Error bars: %s, %s" %(round(min_uncertain,2), round(max_uncertain,2))
    
                # print(min_uncertain, max_uncertain)
    
                axs[0].axvline( min_uncertain, color='gray', linestyle='--', alpha=0.5 )
                axs[0].axvline( max_uncertain, color='gray', linestyle='--', alpha=0.5 )
                        
            plt.tight_layout()
            plt.savefig('alpha_fig_test_%s.pdf' %snap2z[snap])

        if (UNCERTAIN):
            width = min_disp * 1.05

            within_uncertainty = alphas[ (disps < width) ]

            min_uncertain = within_uncertainty[0]
            max_uncertain = within_uncertainty[-1]

            try:
                currentSimGroup_uncertainty.create_dataset( 'snap%s_uncertainty' %snap, data = (min_uncertain,max_uncertain) )
            except:
                print('#'*100)
                print('Not adding uncertainty')
                print('#'*100)
                
            if (DECREMENT):
                start_disp = disps[0]
                min_disp   = disps[min_index]
                                
                try:
                    currentSimGroup_decrement.create_dataset( 'snap%s_decrement' %snap, data=(start_disp, min_disp) )
                except:
                    print('#'*100)
                    print('Not making a decrement catalog')
                    print('#'*100)
        print('')
        
    n_gal = np.array(n_gal)

    print(n_gal)
    print(tot_n_gal)
    
def do_all(dual_criteria):
    
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
    used_alphas = np.ones(len(alphas)) * np.nan

    disps = np.ones(len(alphas)) * np.nan

    best_mu   = None
    best_line = None
    min_alpha = -1.0
    min_disp  = np.inf
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        muCurrent  = star_mass - alpha*np.log10( SFR )

        mu_fit = muCurrent
        Z_fit  = Z_use

        if (FIT_WITH_LOW_REDSHIFT_ONLY):
            low_redshift_mask = (redshifts < 3)

            mu_fit = mu_fit[low_redshift_mask]
            Z_fit  =  Z_fit[low_redshift_mask]
        
        popt = np.polyfit(mu_fit, Z_fit, 1)
        # popt, pcov = curve_fit(line, muCurrent, Z_use)
        
        interp = np.polyval( popt, mu_fit )
        
        if dual_criteria:
            resids = np.sum( np.abs(Z_fit) - np.abs(interp) ) 
        else:
            resids = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
        
        disps[index] = resids
        used_alphas[index] = alpha

        ####################### MAKE GIF #######################
        if (GIF):
            plt.clf()

            fig, ax = plt.subplots(1,2,figsize=(10,6))

            ax1 = ax[0]
            ax2 = ax[1]

            ax1.hist2d(muCurrent, Z_use, bins=(100,100), norm=LogNorm())
                            
            ax1.plot(muCurrent, interp, color='k')
            # ax1.set_xlim(8.0,11.0) 

            ax1.set_xlabel(r'$\mu = \log M_* - %s \log {\rm SFR}$' %round(alpha,2) )
            if (STARS_OR_GAS == "GAS"):
                ax1.set_ylabel(r'$\log {\rm O/H} + 12$ (dex)')
            elif (STARS_OR_GAS == "STARS"):
                ax1.set_ylabel(r'$\log Z_*/Z_\odot$')

            ax2.text(0.5,0.5 ,r'{\rm %s}' %(WHICH_SIM),  transform=ax2.transAxes)
            ax2.text(0.5,0.45,r'${\rm All} ~z$',  transform=ax2.transAxes)
                
            ax2.plot( used_alphas, disps  )

            # ax2.set_xlim(0,1)
            
            ax2.set_xlabel(r'$\alpha$')
            ax2.set_ylabel(r'${\rm Dispersion}\ {\rm from}\ {\rm Median}$')

            plt.tight_layout()
            plt.savefig('./blue_FMR/alpha_gif/%03d' %index)
        ####################### MAKE GIF #######################   

        if (np.nansum(resids) < min_disp):
            min_disp  = np.nansum(resids)
            min_alpha = round(alpha,2)
            best_mu   = muCurrent
            best_line = np.polyval( popt, muCurrent )

    print('%s: best alpha_%s_full = %s' %(WHICH_SIM, STARS_OR_GAS.lower(), min_alpha))    
    if (PLOT):
        plt.clf()
        fig, axs = plt.subplots(2,1,figsize=(5,8))
        
        axs[0].plot( alphas, disps, lw=2 )
        axs[0].axhline( min_disp, color='r', linestyle='--' )
        print('Minimum dispersion: %s' %min_disp)
        axs[0].axvline( min_alpha, color='r',linestyle='--' )
        axs[0].set_xlim(0,1)
        axs[0].set_ylabel(r'${\rm Dispersion}$')
        axs[0].set_xlabel(r'$\alpha$')
        
        
        WHICH_SIM_TEX = {
            "TNG":r"${\rm TNG}$",
            "ORIGINAL":r"${\rm Illustris}$",
            "EAGLE":r"${\rm EAGLE}$"
        }
        
        axs[0].text( 0.4, 0.6 , r'%s' %WHICH_SIM_TEX[WHICH_SIM], transform=axs[0].transAxes )
        axs[0].text( 0.4, 0.5, r'${\rm All}~z$', transform=axs[0].transAxes )

        if (UNCERTAIN):
            width = min_disp * 1.01

            within_uncertainty = alphas[ (disps < width) ]

            min_uncertain = within_uncertainty[0]
            max_uncertain = within_uncertainty[-1]

            axs[0].axvline(min_uncertain, color='gray', linestyle='--')
            axs[0].axvline(max_uncertain, color='gray', linestyle='--')
        
        axs[1].hist2d( best_mu, Z_use, bins=(100,100), norm=LogNorm() )
        axs[1].plot( best_mu, best_line, color='k', lw=2 )
        if (STARS_OR_GAS == "GAS"):
            axs[1].set_ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
        elif (STARS_OR_GAS == "STARS"):
            axs[1].set_ylabel(r'$\log(Z_* [Z_\odot])$')
        axs[1].set_xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))
        
        axs[1].text( 0.6, 0.2, r'${\rm Best}~ \alpha~{\rm fit}$', transform=axs[1].transAxes )

        xmin, xmax = axs[1].get_xlim()
        ymin, ymax = axs[1].get_ylim()
        
        plt.tight_layout()
        
        plt.savefig('full_alpha_%s.pdf' %WHICH_SIM)

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

        # Illustris -- 
        # EAGLE     -- 0.25
        # TNG       -- 0.35
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

        plt.plot( best_mu, best_line, color='k', lw=6.0 )

        if (STARS_OR_GAS == "GAS"):
            plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
        elif (STARS_OR_GAS == "STARS"):
            plt.ylabel(r'$\log(Z_* [Z_\odot])$')
        plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        
        plt.text( 0.75, 0.1, r"${\rm All~}z$", transform=plt.gca().transAxes )
        plt.text( 0.05, 0.9, "%s" %WHICH_SIM_TEX[WHICH_SIM], transform=plt.gca().transAxes )
        
        # plt.tight_layout()
        # plt.savefig( 'test_all_z' )

        Hist1, xedges, yedges = np.histogram2d(best_mu,Z_use,bins=(100,100))
        Hist2, _     , _      = np.histogram2d(best_mu,Z_use,bins=[xedges,yedges])
        
        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)
        
        hist = Hist1/Hist2

        mus_list     = []
        offsets_list = []
        smass_list   = []
        gmass_list   = []
        SFR_list     = []
        R_gas_list   = []
        R_star_list  = []
        
        
        ax_x = 0
        ax_y = 4
        
        axInvis = fig.add_subplot( gs[:,3] )
        axInvis.set_visible(False)
        
        small_axs = []
        ylab_flag = True
        
        for index, time in enumerate(unique):
            # fig = plt.figure(figsize=(10,8))
            
            # if (index == 0):
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

            mus, median_Zs, masks = getMedians(current_mu, current_Z, width=0.1, step=0.05, return_masks=True, min_samp=10)

            _smass = [ np.median(current_smass[np.array(mask)]) for mask in masks ]
            _gmass = [ np.median(current_gmass[np.array(mask)]) for mask in masks ]
            _SFR   = [ np.median(current_SFR  [np.array(mask)]) for mask in masks ]
            _Rgas  = [ np.median(current_Rgas [np.array(mask)]) for mask in masks ]
            _Rstar = [ np.median(current_Rstar[np.array(mask)]) for mask in masks ]
            
            Hist1, current_x, current_y = np.histogram2d(current_mu,current_Z,bins=(100,100))
            Hist2, _        , _         = np.histogram2d(current_mu,current_Z,bins=[current_x,current_y])

            Hist1 = np.transpose(Hist1)
            Hist2 = np.transpose(Hist2)
        
            current_hist = Hist1/Hist2

            vmin = 1 - time
            vmax = 9 - time
            
            plt.pcolormesh( current_x, current_y, current_hist, vmin = vmin, vmax = vmax, cmap=CMAP_TO_USE, rasterized=True )
            
            # cbar = plt.colorbar( label=r"${\rm Redshift}$" )
            # cbar.ax.set_yticklabels(np.arange(0,9))

            # print( "Result of KS test: ", ks_2samp( current_Z, Z_use[~mask] ) )

            FMR_line = interp1d(best_mu, best_line)
            
            plt.plot( best_mu, best_line, color='k', lw=4.5 )
            # plt.scatter( mus, median_Zs, color='k' )

            # if (STARS_OR_GAS == "GAS"):
            #     plt.ylabel(r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$')
            # elif (STARS_OR_GAS == "STARS"):
            #     plt.ylabel(r'$\log(Z_* [Z_\odot])$')
            # plt.xlabel(r'$\mu_{%s} = \log M_* - %s\log{\rm SFR}$' %(min_alpha,min_alpha))

            plt.text( 0.65, 0.1, r"$z = %s$" %int(time), transform=plt.gca().transAxes )
            # plt.text( 0.05, 0.9, "%s" %WHICH_SIM_TEX[WHICH_SIM], transform=plt.gca().transAxes )

            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            
            # plt.tight_layout()
            # plt.savefig( 'test_z=%s' %int(time) )

            offsets_list.append( median_Zs - FMR_line(mus) )
            mus_list.append( mus )
            smass_list.append( _smass )
            gmass_list.append( _gmass )
            SFR_list.append( _SFR )
            R_gas_list.append( _Rgas )
            R_star_list.append( _Rstar )
            

        plt.tight_layout()
        if dual_criteria:
            plt.savefig( BLUE + 'JWST/' + '%s_big_fig_dc.pdf' %WHICH_SIM, bbox_inches='tight' )
        else:
            plt.savefig( BLUE + 'JWST/' + '%s_big_fig.pdf' %WHICH_SIM, bbox_inches='tight' )
        plt.clf()

        mpl.rcParams['font.size'] = fs_og
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.major.width'] = 1.5
        mpl.rcParams['xtick.minor.width'] = 1.0
        mpl.rcParams['ytick.minor.width'] = 1.0
        mpl.rcParams['xtick.major.size']  = 7.5
        mpl.rcParams['ytick.major.size']  = 7.5
        mpl.rcParams['xtick.minor.size']  = 3.5
        mpl.rcParams['ytick.minor.size']  = 3.5
        mpl.rcParams['axes.linewidth']    = 2.25
        
        fig, axs = plt.subplots(2, 3, figsize=(20,12))
        cmap = plt.cm.RdBu
        vmin = -0.2#min([min(sublist) for sublist in offsets_list])
        vmax = +0.2#max([max(sublist) for sublist in offsets_list])
        N_LEVELS = np.linspace(min([min(sublist) for sublist in offsets_list]), max([max(sublist) for sublist in offsets_list]), 100)
        print "Starting mu plot\n"

        all_mus = []
        all_redshifts = []
        all_offsets = []
        for index, offset in enumerate(offsets_list):
            mu       = list(mus_list[index])
            redshift = list(np.ones( len(mus_list[index]) ) * index)
            
            all_mus       += mu
            all_redshifts += redshift
            all_offsets   += list(offset)
            
            # avgOffset = round(np.median(offset),2)
            
        axs[0,0].tricontourf( all_redshifts, all_mus, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )
        for index, offset in enumerate(offsets_list):
            mu       = list(mus_list[index])
            redshift = list(np.ones( len(mus_list[index]) ) * index)
            
            # avgOffset = round(np.median(offset),2)
            
            axs[0,0].scatter( redshift, mu, c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[0,0].set_xlabel( r"${\rm Redshift}$" )
        axs[0,0].set_ylabel( r"$\mu_{%s} = \log M_* - %s \log {\rm SFR}$" %(min_alpha,min_alpha) )

        #axs[0].text( 0.8, 0.9, WHICH_SIM_TEX[WHICH_SIM], transform=axs[0].transAxes )
        #plt.colorbar( label=r'${\rm Metallicity~Offset~(dex)}$' )
        #plt.savefig( 'offsets_%s_mu.pdf' %WHICH_SIM )

        all_masses    = []
        all_redshifts = []
        all_offsets   = []
        print "Starting stellar mass plot\n"
        for index, offset in enumerate(offsets_list):
            _smass   = list(np.array(smass_list[index]) )
            redshift = list(np.ones( len(_smass) ) * index )

            all_masses    += _smass
            all_redshifts += redshift
            all_offsets   += list(offset)

        axs[0,1].tricontourf( all_redshifts, all_masses, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )
        for index, offset in enumerate(offsets_list):
            _smass   = list(np.array(smass_list[index]) )
            redshift = list(np.ones( len(_smass) ) * index )

            axs[0,1].scatter( redshift, _smass, c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[0,1].set_xlabel( r"${\rm Redshift}$" )
        axs[0,1].set_ylabel( r"$\log M_*~[M_\odot]$" )

        axs[0,1].text( 0.75, 0.9, WHICH_SIM_TEX[WHICH_SIM], transform=axs[0,1].transAxes, fontsize=24 )
        #plt.colorbar( label=r'${\rm Metallicity~Offset~(dex)}$' )
        #plt.savefig( 'offsets_%s_mass.pdf' %WHICH_SIM )

        all_SFRs      = []
        all_redshifts = []
        all_offsets   = []
        print "Starting SFR plot\n"
        for index, offset in enumerate(offsets_list):
            _SFR     = list(np.log10(SFR_list[index]))
            redshift = list(np.ones( len(_SFR) ) * index)

            all_SFRs      += _SFR
            all_redshifts += redshift
            all_offsets   += list(offset)
        im = axs[0,2].tricontourf( all_redshifts, all_SFRs, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )    
        for index, offset in enumerate(offsets_list):
            _SFR     = list(np.log10(SFR_list[index]))
            redshift = list(np.ones( len(_SFR) ) * index)

            axs[0,2].scatter( redshift, _SFR, c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[0,2].set_xlabel( r"${\rm Redshift}$" )
        axs[0,2].set_ylabel( r"${\rm SFR}~[\log (M_\odot/{\rm yr})]$" )

        all_times     = []
        all_redshifts = []
        all_offsets   = []
        print "Starting Mgas/SFR plot\n"
        for index, offset in enumerate(offsets_list):
            _gmass   = np.array(gmass_list[index]) 
            _SFR     = np.array(SFR_list[index])
            redshift = list(np.ones( len(_gmass) ) * index )

            all_times     += list( np.log10(_gmass / _SFR) )
            all_redshifts += redshift
            all_offsets   += list(offset)
        
        axs[1,0].tricontourf( all_redshifts, all_times, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )
        for index, offset in enumerate(offsets_list):
            _gmass   = np.array(gmass_list[index]) 
            _SFR     = np.array(SFR_list[index])
            redshift = list(np.ones( len(_gmass) ) * index )

            axs[1,0].scatter( redshift, np.log10(_gmass / _SFR), c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[1,0].set_xlabel( r"${\rm Redshift}$" )
        axs[1,0].set_ylabel( r"$\log M_{\rm gas}/{\rm SFR}~[{\rm yr}]$" )

        all_Rs        = []
        all_redshifts = []
        all_offsets   = []
        print "Starting Rgas plot\n"
        for index, offset in enumerate(offsets_list):
            _Rgas    = list(np.array(R_gas_list[index]))
            redshift = list(np.ones( len(_Rgas) ) * index )

            all_Rs        += _Rgas
            all_redshifts += redshift
            all_offsets   += list(offset)

        axs[1,1].tricontourf( all_redshifts, all_Rs, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )
        for index, offset in enumerate(offsets_list):
            _Rgas    = list(np.array(R_gas_list[index]))
            redshift = list(np.ones( len(_Rgas) ) * index )

            axs[1,1].scatter( redshift, _Rgas, c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[1,1].set_xlabel( r"${\rm Redshift}$" )
        axs[1,1].set_ylabel( r"$R_{\rm gas}~[{\rm kpc}]$" )

        all_Rs        = []
        all_redshifts = []
        all_offsets   = []
        print "Starting Rstar plot\n"
        for index, offset in enumerate(offsets_list):
            _Rstar   = list(np.array(R_star_list[index]))
            redshift = list(np.ones( len(_Rstar) ) * index )

            all_Rs        += _Rstar
            all_redshifts += redshift
            all_offsets   += list(offset)

        axs[1,2].tricontourf( all_redshifts, all_Rs, all_offsets, N_LEVELS, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax )
        for index, offset in enumerate(offsets_list):
            _Rstar   = list(np.array(R_star_list[index]))
            redshift = list(np.ones( len(_Rstar) ) * index )

            axs[1,2].scatter( redshift, _Rstar, c=offset, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, edgecolors='gray' )

        axs[1,2].set_xlabel( r"${\rm Redshift}$" )
        axs[1,2].set_ylabel( r"$R_{\rm star}~[{\rm kpc}]$" )
        
        #plt.text( 0.8, 0.9, WHICH_SIM_TEX[WHICH_SIM], transform=plt.gca().transAxes )
        max_round = round( max([max(sublist) for sublist in offsets_list]),1 )
        min_round = round( min([min(sublist) for sublist in offsets_list]),1 )
        cbar = plt.colorbar( mappable=im, label=r'${\rm Metallicity~Offset~(dex)}$', ticks=np.arange( min_round, max_round, 0.1 ) )
        plt.tight_layout()
        plt.savefig( 'offsets_%s.pdf' %WHICH_SIM )
        
def getMedians(x,y,width=0.1,step=0.05,return_masks=False,percentile=50,min_samp=10):
    start = np.min(x)
    end   = np.max(x)
    
    current = start
    
    medians = []
    xs      = []
    if (return_masks):
        masks = []
    
    while (current < end + 2*step):
        
        mask = ((x > (current)) & (x < (current + width)))
        if (return_masks):
            masks.append( mask )
        
        if (len(y[mask]) > min_samp):
            medians.append( np.percentile( y[mask], percentile ) )
        else:
            medians.append( np.nan )
            
        xs.append( current )
    
        current += step
    
    medians = np.array(medians)
    xs      = np.array(xs)
    
    nonans = ~(np.isnan(medians))
    
    xs      = xs[nonans] 
    medians = medians[nonans]

    if (return_masks):
        masks = np.array(masks)
        masks = masks[nonans]
        masks = list(masks)
        return xs, medians, masks
    else:
        return xs, medians
    
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
    
if (COMBINE_ALL):
    do_all(DUAL_CRITERIA)
else:
    do()
alpha_file.close()
uncertain.close()
if (DECREMENT):
    decrement_file.close()