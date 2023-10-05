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

def do(sim,full_alpha,full_slope,full_intercept,dual_criteria):
    
    fig, axs = plt.subplots( 3,4, figsize=(10,10),
                             gridspec_kw={'width_ratios': [1, 1, 0.4, 1]} )
    
    do_which_snaps = ['z=0','z=4','z=8']
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    redshifts = np.arange(0,9)
    
    newcolors   = plt.cm.rainbow(np.linspace(0, 1, len(redshifts)-1))
    CMAP_TO_USE = ListedColormap(newcolors)
    
    global_mappable = None
    
    ax_idx = -1
    
    for snap_index, snap in enumerate(snapshots):
        if (snap2z[snap] not in do_which_snaps):
            continue
        else:
            ax_idx += 1
        
        ax_row = axs[ax_idx,:]
        
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
            # interp = popt[0] * muCurrent + popt[1]

            if dual_criteria:
                disp[index] = np.sum( np.abs(Z_use) - np.abs(interp) )# np.std( np.abs(Z_use) - np.abs(interp) ) * np.sum( np.abs(Z_use) - np.abs(interp) )
            else:
                disp[index] = np.std( np.abs(Z_use) - np.abs(interp) ) 

        argmin = np.argmin(disp)

        # Which ax is which
        ax_global = ax_row[0]
        ax_dist   = ax_row[3]
        ax_local  = ax_row[1]
        ax_row[2].axis("off")
        
        # Plot the global alpha
        global_mu   = star_mass - full_alpha * np.log10(SFR)
        global_line = full_slope * global_mu + full_intercept
        
        plot_global_mu   = np.linspace(7,11,200)
        plot_global_line = full_slope * plot_global_mu + full_intercept
        
        Hist1, current_x, current_y = np.histogram2d(global_mu,Z_use,bins=(100,100))
        Hist2, _        , _         = np.histogram2d(global_mu,Z_use,bins=[current_x,current_y])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        current_hist = Hist1/Hist2 - 1

        vmin = 0 - redshifts[snap_index]
        vmax = 8 - redshifts[snap_index]

        ax_global.pcolormesh( current_x, current_y, current_hist, 
                              vmin = vmin, vmax = vmax, cmap=CMAP_TO_USE,
                              rasterized=True )
        
        xmin,xmax = ax_global.get_xlim()
        ymin,ymax = ax_global.get_ylim()
        ax_global.plot( plot_global_mu, plot_global_line, color='k', lw=2.5 )
        ax_global.set_xlim(xmin,xmax)
        ax_global.set_ylim(ymin,ymax)
        
        ax_global.text( 0.05, 0.85, r"$z=%s$" %int(snap_index), transform=ax_global.transAxes )
        ax_global.text( 0.7 , 0.2 , r"${\rm FMR}$", transform=ax_global.transAxes, ha='center')
        ax_global.text( 0.7 , 0.1 , r"$\alpha_{\rm min} = %s$" %full_alpha,
                        transform=ax_global.transAxes, ha='center', fontsize=fs_og*0.75)
        
        ax_global.set_xlabel( r"$\mu_{%s}$" %full_alpha, fontsize=fs_og*0.75 )
        
        global_ymin, global_ymax = ax_global.get_ylim()
        global_xmin, global_xmax = ax_global.get_xlim()
        
        if (sim == 'ORIGINAL') and (snap2z[snap] == 'z=4'):
            ax_global.set_xticklabels([7,8,9,''])
        
        # Plot the individual alpha
        best_alpha = round( alphas[argmin], 2 )
        best_mu    = star_mass - best_alpha * np.log10(SFR)
        best_line  = a_s[argmin] * best_mu + b_s[argmin]
        plot_best_line = a_s[argmin] * np.linspace(0,10) + b_s[argmin]
        
        Hist1, current_x, current_y = np.histogram2d(best_mu,Z_use,bins=(100,100))
        Hist2, _        , _         = np.histogram2d(best_mu,Z_use,bins=[current_x,current_y])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        current_hist = Hist1/Hist2 - 1

        vmin = 0 - redshifts[snap_index]
        vmax = 8 - redshifts[snap_index]

        mappable = ax_local.pcolormesh( current_x, current_y, current_hist,
                                        vmin = vmin, vmax = vmax, cmap=CMAP_TO_USE,
                                        rasterized=True )
        
        if (snap_index == 0):
            global_mappable = mappable

        ax_local.plot( best_mu, best_line, color='k', lw=2.5 )
        
        ax_local.text( 0.7, 0.2, r"${\rm RSZR}$", transform=ax_local.transAxes, ha='center')
        ax_local.text( 0.7, 0.1, r"$\alpha_{\rm min} = %s$" %best_alpha,
                       transform=ax_local.transAxes, ha='center', fontsize=fs_og*0.75 )
        
        ax_local.set_xlabel( r"$\mu_{%s}$" %best_alpha, fontsize=fs_og*0.75 )
        
        ax_local.set_ylim( global_ymin, global_ymax )
        
        local_xmin, local_xmax = ax_local.get_xlim()
        
        xmin, xmax = min( global_xmin, local_xmin ), max( global_xmax, local_xmax )
        ax_local .set_xlim( xmin, xmax )
        ax_global.set_xlim( xmin, xmax )
        
        ax_local.set_yticklabels([])
        
        # Plot the offsets
        offsets_global = Z_use - global_line
        offsets_local  = Z_use - best_line
        
        glob_col = 'brown'
        loc_col  = 'k'
        
        nbins = 50
        if (snap_index > 7):
            nbins = 25
        
        ax_dist.hist( offsets_global, color=glob_col, alpha=0.25, bins=nbins )
        ax_dist.hist( offsets_local , color=loc_col , alpha=0.25, bins=nbins )
        
        ax_dist.set_xlim( -0.65,0.65 )
        
        ax_dist.axvline( 0, color='b', lw=1.5 )
        
        ax_dist.set_xlabel( r'${\rm Offsets}~({\rm dex})$',
                            fontsize=fs_og*0.75 )
        
        ax_dist.text( 0.05,0.85 , r'${\rm FMR}$', color=glob_col,
                      transform=ax_dist.transAxes )
        ax_dist.text( 0.05,0.775, r'$\sigma=%.3f$' %np.std( offsets_global ), color=glob_col,
                      transform=ax_dist.transAxes, fontsize=fs_og*0.75 )
        ax_dist.text( 0.05,0.65 , r'${\rm RSZR}$' , color=loc_col ,
                      transform=ax_dist.transAxes )
        ax_dist.text( 0.05,0.575, r'$\sigma=%.3f$' %np.std( offsets_local ), color=loc_col,
                      transform=ax_dist.transAxes, fontsize=fs_og*0.75  )

        ax_dist.set_ylabel( r'${\rm Counts}$' )
        
    for ax in axs[:,0]:
        ax.set_ylabel( r'$\log {\rm (O/H)} + 12~({\rm dex})$', fontsize=fs_og*0.75 )
    # for ax in axs[:,-1]:
    #     ax.set_ylabel( r'$\log {\rm (O/H)} + 12~({\rm dex})$', fontsize=fs_og*0.75 )
    
    global_xmin, global_xmax = 99999, 7.5
    global_ymin, global_ymin = 99999, 7.5
    
    ax_fits = list(np.concatenate( (np.array(axs[:,0]), np.array(axs[:,1])) ))
    
    for ax in ax_fits:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            global_xmin, global_xmax = min( global_xmin, xmin ), max( global_xmax, xmax )
            global_ymin, global_ymax = min( global_ymin, ymin ), max( global_ymax, ymax )
            
    for ax in ax_fits:
            ax.set_xlim( global_xmin, global_xmax )
            ax.set_ylim( global_ymin, global_ymax )
    
    axs[0,0].set_title( whichSim2Tex[sim], color='white' )
    axs[0,0].text( 1.0, 1.05, whichSim2Tex[sim], ha='center', transform=axs[0,0].transAxes )
    
    # cbar_ax = fig.add_axes([0.15, 1.01, 0.7, 0.02])
    # cbar = fig.colorbar( global_mappable, cax=cbar_ax, orientation='horizontal' )
    # cbar.ax.set_yticklabels(redshifts)
    # cbar.ax.set_title(r'${\rm Redshift}$')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0.25)
    
    if dual_criteria:
        plt.savefig( BLUE + 'JWST/' + '%s_alphas_comparison_dc' %sim + '.pdf', bbox_inches='tight' )
    else:
        plt.savefig( BLUE + 'JWST/' + '%s_alphas_comparison' %sim + '.pdf', bbox_inches='tight' )
    return global_mappable

def get_full_alpha(sim, dual_criteria):
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
        if dual_criteria:
            disps[index] =  np.sum( np.abs(Z_fit) - np.abs(interp) )#np.std( np.abs(Z_fit) - np.abs(interp) ) * np.sum( np.abs(Z_fit) - np.abs(interp) )
        else:
            disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 
        
    argmin = np.argmin(disps)

    return round( alphas[argmin], 2 ), a_s[argmin], b_s[argmin]

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

sims = ['original','tng','eagle']

global_mappable = None

dual_criteria = False

for index, sim in enumerate(sims):
    sim = sim.upper()
    print(sim)
    full_alpha, full_slope, full_intercept = get_full_alpha(sim, dual_criteria)
    global_mappable = do(sim,full_alpha, full_slope, full_intercept, dual_criteria)
