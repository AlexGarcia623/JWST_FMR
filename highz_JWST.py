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

mpl.rcParams['text.usetex']        = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

mpl.rcParams['font.size'] = 25
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

BLUE_DIR = BLUE + WHICH_SIM + "/"

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

##### PLUS ONE BECAUE OF BOXPLOT PLOTTING 1-9 and not 0-8!!! #####
# From Langeroodi & Hjorth 2023
Langeroodi23 = [
    (3.335522182113861 +1, -0.20249725880570124),
    (3.6810974633172764+1, -0.2148973130135885),
    (4.895083098227156 +1, -0.30434032697211966),
    (6.884266160726384 +1, -0.37892545183505977),
]

Langeroodi23_up = [
    (3.4450344343283765+1, -0.19567815298944136),
    (3.8496347127597974+1, -0.2145277137823549),
    (5.74612228806564  +1, -0.2893161182224735),
    (7.920770244797889 +1, -0.3766524165629732),
]

Langeroodi23_down = [
    (3.200655422636721 +1, -0.19621407187473006),
    (3.5884019761238894+1, -0.21510059259076697),
    (4.473666054774606 +1, -0.2921065924182873),
    (6.100567950818661 +1, -0.38064408826029594),
]

Langeroodi23_yup = [
    (3.334746023728271 +1, -0.06434106617058188),
    (3.6719683623058064+1, -0.08991733297194693),
    (4.893678621148467 +1, -0.05434340696571316),
    (6.8907341472729735+1, -0.030227057127721002),
]

Langeroodi23_ydown = [
    (3.3375180179625223+1, -0.557756039867437),
    (3.6737793985388505+1, -0.412281782453892),
    (4.89711589399894  +1, -0.6661779743498129),
    (6.885670637805073 +1, -0.628922371841466),

]

# From Curti+2023
Curti23 = [
    (4.248366013071896+1, -0.2290322580645161),
    (5.76470588235294 +1, -0.3451612903225807),
    (7.66013071895425 +1, -0.5096774193548388),
]

Curti23_up = [
    (4.77124183006536  +1, -0.22580645161290325),
    (6.2483660130718945+1, -0.3451612903225807),
    (8.19607843137255  +1, -0.5096774193548388),
]

Curti23_down = [
    (3.725490196078432 +1, -0.22580645161290325),
    (5.267973856209151 +1, -0.3451612903225807),
    (7.1111111111111125+1, -0.5096774193548388),
]

Curti23_yup = [
    (4.248366013071896+1, 0.02580645161290318),
    (5.76470588235294 +1, -0.08064516129032262),
    (7.66013071895425 +1, -0.11935483870967745),
]

Curti23_ydown = [
    (4.248366013071896+1, -0.4774193548387098),
    (5.76470588235294 +1, -0.6129032258064517),
    (7.66013071895425 +1, -0.9032258064516132),
]

# From Nakajima+2023
## Curti+2020 alpha = 0.66
Nakajima23_C20 = [
    (4.970986707566462+1, -0.21710974778459424),
    (6.856680299931832+1, -0.2440567484662577),
    (8.490030674846626+1, -0.46206117927743695),
]

Nakajima23_C20_up = [
    (5.493311179277437+1, -0.21958077709611445),
    (7.557302317655076+1, -0.24288513974096793),
    (8.694146216768914+1, -0.4851312201772322),
]

Nakajima23_C20_down = [
    (4.448747443762782+1, -0.2213275391956373),
    (6.168754260395365+1, -0.24186264485344244),
    (8.286468984321743+1, -0.4824684730743012),
]

Nakajima23_C20_yup = [
    (4.980487389229721+1, 0.037086741649625),
    (6.866564417177915+1, -0.019959952283571836),
    (8.488113496932517+1, -0.31156271301976823),
]

Nakajima23_C20_ydown = [
    (4.987048398091343+1, -0.47795245398773),
    (6.859534764826175+1, -0.46813224267212017),
    (8.492118268575325+1, -0.6259372869802315),
]

## Curti+2020 alpha = 0.66
Nakajima23_AM13 = [
    (4.96358326014583+1, -0.07978799452625562),
    (6.838463266681646+1, -0.0701885173913932),
    (8.466820530626418+1, -0.33876963297318285),
]

Nakajima23_AM13_up = [
    (5.476072792630869+1, -0.07895059332938459),
    (7.550948714282797+1, -0.06902432548354809),
    (8.666898143420273+1, -0.3449786564816897),
]

Nakajima23_AM13_down = [
    (4.438634831804906+1, -0.08391372725230295),
    (6.163517902003636+1, -0.07455934315067103),
    (8.26694716202692 +1, -0.3489001450133782),
]

Nakajima23_AM13_yup = [
    (4.96031535303609  +1, 0.18164457425297686),
    (6.8480627438165085+1, 0.1618533118196115),
    (8.46457384448847  +1, -0.15903474193746048),
]

Nakajima23_AM13_ydown = [
    (4.954351422560813+1, -0.34124098772492406),
    (6.853904127775169+1, -0.30545740487326656),
    (8.481362717264762+1, -0.5021445640407671),
]

def do(ax,sim,c,alpha,a,b,do_FMR):
    
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_offsets = []
    means       = []
    
    z0_alpha = 0.0
    z0_a     = 1.0
    z0_b     = 0.0
    
    if do_FMR:
        z0_alpha = alpha
        z0_a     = a
        z0_b     = b
    
    for snap in snapshots:
        
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
            
            if not (do_FMR):
                z0_alpha = round( alphas[argmin], 2 )
                z0_a     = a_s[argmin]
                z0_b     = b_s[argmin]
            
        mu = star_mass - z0_alpha * np.log10(SFR)
        
        z0_FMR_Z_predictions = z0_a * mu + z0_b
        
        offsets = Z_use - z0_FMR_Z_predictions
        
        all_offsets.append( offsets )
        means.append( np.median(offsets) )
        
    bp = ax.boxplot( all_offsets, patch_artist=True,
                     whiskerprops = dict(color = c,alpha=0.5),
                     capprops     = dict(color = c,alpha=0.5),
                     boxprops     = dict(facecolor = 'white',color=c,alpha=0.5),
                     flierprops   = dict(marker='+', alpha=0.25,markersize=2.5,markerfacecolor=c,
                                         markeredgecolor=c),
                     widths       = np.ones(len(means)) * 0.25
              )
    for median in bp['medians']:
        median.set_color(color)
        
    for index, coords in enumerate(Langeroodi23):
        x = coords[0]
        y = coords[1]
        x_err_up   = Langeroodi23_up[index][0] - x
        x_err_down = x - Langeroodi23_down[index][0]
        y_err_up   = Langeroodi23_yup[index][1] - y
        y_err_down = y - Langeroodi23_ydown[index][1]
        
        x_err = np.array([ [x_err_down, x_err_up] ]).T
        y_err = np.array([ [y_err_down, y_err_up] ]).T
        if (index == 0):
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='purple', marker='s', markersize=8,
                       label = r'${\rm Langeroodi\;\&\;Hjorth\;(2023)}$' )
        else:
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='purple', marker='s', markersize=8 )
            
    for index, coords in enumerate(Curti23):
        x = coords[0]
        y = coords[1]
        x_err_up   = Curti23_up[index][0] - x
        x_err_down = x - Curti23_down[index][0]
        y_err_up   = Curti23_yup[index][1] - y
        y_err_down = y - Curti23_ydown[index][1]
        
        x_err = np.array([ [x_err_down, x_err_up] ]).T
        y_err = np.array([ [y_err_down, y_err_up] ]).T
        if (index == 0):
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='violet', marker='^', markersize=8,
                       label = r'${\rm Curti+(2023)}$' )
        else:
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='violet', marker='^', markersize=8 )
            
    for index, coords in enumerate(Nakajima23_C20):
        x = coords[0]
        y = coords[1]
        x_err_up   = Nakajima23_C20_up[index][0] - x
        x_err_down = x - Nakajima23_C20_down[index][0]
        y_err_up   = Nakajima23_C20_yup[index][1] - y
        y_err_down = y - Nakajima23_C20_ydown[index][1]
        
        x_err = np.array([ [x_err_down, x_err_up] ]).T
        y_err = np.array([ [y_err_down, y_err_up] ]).T
        if (index == 0):
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='goldenrod', marker='*', markersize=10,
                       label = r'${\rm Nakajima+(2023; C20)}$' )
        else:
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='goldenrod', marker='*', markersize=10 )
            
    for index, coords in enumerate(Nakajima23_AM13):
        x = Nakajima23_C20[index][0]#coords[0]
        y = coords[1]
        x_err_up   = Nakajima23_C20_up[index][0] - x
        x_err_down = x - Nakajima23_C20_down[index][0]
        y_err_up   = Nakajima23_AM13_yup[index][1] - y
        y_err_down = y - Nakajima23_AM13_ydown[index][1]
        
        x_err = np.array([ [x_err_down, x_err_up] ]).T
        y_err = np.array([ [y_err_down, y_err_up] ]).T
        if (index == 0):
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='navy', marker='o', markersize=8,
                       label = r'${\rm Nakajima+(2023; AM13)}$' )
        else:
            ax.errorbar( x, y, xerr=x_err, yerr=y_err, color='navy', marker='o', markersize=8 )
            
            
    ax.set_xticklabels( np.arange(0,9) )
    
    redshifts = np.arange(0,9) + 1
    
    # ax.scatter( redshifts, means, color='k', marker='x', s=100 )
    
    popt   = np.polyfit( redshifts, means, 1 )
    interp = np.polyval( popt, redshifts )
    
    ax.plot( redshifts, interp, color='k', lw=2.5, linestyle='--' )
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.text( 0.05, 0.85, whichSim2Tex[sim], transform=ax.transAxes, color=color )

def get_FMR(sim):
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
    
    a_s, b_s = np.ones( len(alphas) ), np.ones( len(alphas) )
    
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


fig,axs = plt.subplots(3,1,figsize=(8,13),sharex=True, sharey=True)

sims   = ['ORIGINAL','TNG','EAGLE']
cols   = ['C1','C2','C0']

do_FMR  = True
do_LFMR = False # This is not an option I should use!!!

if not do_FMR and do_LFMR:
    do_LFMR = False
    print('')
    print('#'*100)
    print('Not doing local FMR because we are not doing the FMR')
    print('#'*100)
    print('')

for index, sim in enumerate(sims):
    ax = axs[index]
    color = cols[index]#'k'#'C' + str(index)
    if (do_FMR):
        alpha, a, b = get_FMR(sim)
    else:
        alpha, a, b = 0.0, 0.0, 0.0
    do(ax, sim, color, alpha, a, b, do_FMR)

ymin, ymax = axs[0].get_ylim()
axs[0].set_ylim( ymin, ymax*1.2 )
    
axs[0].set_xlim(0,10)
    
leg = axs[1].legend( loc='upper right', frameon=False, fontsize=18,
                     handlelength=0, labelspacing=0.05 )
colors = ['purple','violet','goldenrod','navy']
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
if do_FMR:
    axs[1].set_ylabel( r'$\log {\rm (O/H)} - \log{\rm (O/H)}_{\rm GFMR}$' )
else:
    axs[1].set_ylabel( r'$\log {\rm (O/H)} - \log{\rm (O/H)}_{{\rm LFMR}}$' )
    
axs[2].set_xlabel( r'${\rm Redshift}$' )

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)

if do_FMR:
    plt.savefig( BLUE + 'JWST/' + 'all_offsets_FMR' + '.pdf', bbox_inches='tight' )
else:
    plt.savefig( BLUE + 'JWST/' + 'all_offsets' + '.pdf', bbox_inches='tight' )