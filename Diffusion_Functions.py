# Diffusion_Functions
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Model, Parameters, fit_report
import matplotlib.pyplot as plt

import scipy.optimize
from sklearn.metrics import r2_score

lw = 2
fontsize = 24
ticklen = lw*5

params = {'legend.fontsize': 'large',
            'font.family': 'sans-serif',
            'font.sans-serif': 'Tahoma',
            'font.weight': 'regular',
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'xtick.labelsize': fontsize,
            'xtick.major.width': lw,
            'xtick.major.size': ticklen,
            'xtick.minor.visible': True,
            'xtick.minor.width': 3/4*lw,
            'xtick.minor.size': ticklen/2,
            'ytick.labelsize': fontsize,
            'ytick.major.width': lw,
            'ytick.major.size': ticklen,
            'ytick.minor.visible': True,
            'ytick.minor.width': 3/4*lw,
            'ytick.minor.size': ticklen/2,
            'axes.labelweight': 'regular',
            'axes.linewidth': lw,
            'axes.titlepad': 25,
            'xtick.direction': 'in',
            'ytick.direction': 'in'}
plt.rcParams.update(params)


# load 'xf2'
def xf2(datapath, procno=1, mass=1, f2l=10, f2r=0):
    """xAxppm, real_spectrum, expt_parameters = xf2(datapath, procno=1, mass=1, f2l=10, f2r=0)
    """
    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2rr")
    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")
    proc2s = os.path.join(datapath,"pdata",str(procno),"proc2s")
    acqu2s = os.path.join(datapath,"acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    ####################################

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    Lstr = "##$L= (0..31)"
    CNSTstr = "##$CNST= (0..63)"
    TDstr = "##$TD= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    L1 = float("NaN")
    L2 = float("NaN")
    CNST31 = float("NaN")
    TD = float("NaN")

    with open(acqus,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if TDstr in line.decode():
                linestr = line.decode()
                TD = float(linestr.strip(TDstr))
            if Lstr in line.decode():
                line = next(input)
                linestr = line.decode()
                L = (linestr.strip("\n").split(" "))
                L1 = float(L[1])
                L2 = float(L[2])
            if CNSTstr in line.decode():
                CNST = []
                line = next(input)
                while "##$CPDPRG=" not in str(line):
                    linestr = line.decode()
                    CNST.extend(linestr.strip("\n").split(" "))
                    line = next(input)
                CNST31 = float(CNST[31])
            if ~np.isnan(O1) and ~np.isnan(OBS) and ~np.isnan(L1) and ~np.isnan(TD) and ~np.isnan(CNST31) and not len(NUC)==0:
                break

    ####################################

    # Get procs

    SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '
    XDIM_F2str = '##$XDIM= '

    SW = float("NaN")
    SI = int(0)
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)


    with open(procs,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = int(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str):len(linestr)-1])
            if ~np.isnan(SW) and SI!=int(0) and ~np.isnan(NC_proc) and ~np.isnan(SF) and XDIM_F2!=int(0):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    TDstr_2 = "##$TD= "

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    TD_2 = float("NaN")

    with open(acqu2s,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr)-1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr)-1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2):len(linestr)-2])
            if TDstr_2 in line.decode():
                linestr = line.decode()
                TD_2  = float(linestr.strip(TDstr_2))
            if ~np.isnan(O1_2) and ~np.isnan(OBS_2) and ~np.isnan(TD_2) and not len(NUC_2)==0:
                break

    ####################################

    # # Get proc2s for indirect dimension

    SIstr_2 = '##$SI= '
    XDIM_F1str = '##$XDIM= '

    XDIM_F1 = int(0)
    SI_2 = int(0)

    with open(proc2s,"rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = int(linestr[len(SIstr_2):len(linestr)-1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str):len(linestr)-1])
            if SI_2!=int(0) and XDIM_F1!=int(0):
                break

    ####################################

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF

    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4', count=-1)
    if not bool(real_spectrum.any()):
        print(real_spectrum)
        print("Error: Spectrum not read.")
        
    # print(np.shape(real_spectrum),int(XDIM_F1), int(XDIM_F2), int(SI_2), int(SI))
    if XDIM_F1 == 1:
        real_spectrum = real_spectrum.reshape([int(SI_2),int(SI)])
    else:
        # to shape the column matrix according to Bruker's format, matrices are broken into (XDIM_F1,XDIM_F2) submatrices, so reshaping where XDIM_F1!=1 requires this procedure.
        column_matrix = real_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols*submatrix_rows

        blocks = np.array(np.array_split(column_matrix,submatrix_number))  # Split into submatrices
        blocks = np.reshape(blocks,(submatrix_rows,submatrix_cols,-1)) # Reshape these submatrices so each has its own 1D array
        real_spectrum = np.vstack([np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks])  # Concatenate submatrices in the correct orientation

    f2l_temp = max(xAxppm)
    f2r_temp = min(xAxppm)

    if f2l<f2r:
        f2l, f2r = f2r,f2l

    xlow = np.argmax(xAxppm<f2l)
    xhigh = np.argmax(xAxppm<f2r)

    if xlow == 0:
        xlow = np.argmax(xAxppm==f2l_temp)
    if xhigh == 0:
        xhigh = np.argmax(xAxppm==f2r_temp)

    if xlow>xhigh:
        xlow, xhigh = xhigh, xlow
    xAxppm = xAxppm[xlow:xhigh]
    real_spectrum = real_spectrum[:int(SI_2),xlow:xhigh]
    # real_spectrum = real_spectrum[:,xlow:xhigh]

    expt_parameters = {'NUC': NUC, "L1": L1, "CNST31": CNST31, "TD_2": TD_2}
    
    return xAxppm, real_spectrum, expt_parameters

# Gradient Paramaters Import Function
def diff_params_import(datapath, NUC, TD2):
    """
    delta, DELTA, expectedD, Gradlist = diff_params_import(datapath, NUC)
    obtains delta, DELTA, a guess for D, and the gradient list from the diff.xml file
    """
    import xml.etree.ElementTree as ET

    diff_params_path = os.path.join(datapath,"diff.xml")

    tree = ET.parse(diff_params_path)
    root = tree.getroot()

    delta = float(root.find(".//delta").text) # [ms]
    delta = delta/1000 # [s]
    DELTA = float(root.find(".//DELTA").text) # [ms]
    DELTA = DELTA/1000  # [s]
    exD = float(root.find(".//exDiffCoff").text) # [m2/s]
    x_values_element = root.find(".//xValues/List")
    x_values_list = x_values_element.text.split()
    x_values = [float(value) for value in x_values_list]
    Gradlist = x_values[1::4] # [G/cm]
    Gradlist = [x/100 for x in Gradlist] # [T/m]
    Gradlist = Gradlist[:int(TD2)]

    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    # gamma = gamma  # [1/T/s]

    return delta, DELTA, exD, Gradlist, gamma

# Peak picking
def xf2_peak_pick(xAxppm, real_spectrum,height = 0.1, threshold = [0.001, 1], prominence = [0.001, 1], peak_pos = float("NaN"), norm=False, plottype='none'):
    # import math
    # Initial slice
    slice_1 = real_spectrum[0,:]
    # print(real_spectrum)
    min_slice_1 = min(slice_1)
    # slice_1 = slice_1-min_slice_1
    max_slice_1 = max(slice_1)
    slice_1 = slice_1/max_slice_1
    # real_spectrum_norm = real_spectrum-min_slice_1
    real_spectrum_norm = real_spectrum/max_slice_1

    if np.isnan(peak_pos).any():
        pl = find_peaks(slice_1, prominence=prominence)
        # pl = find_peaks(slice_1, height=height, threshold=threshold, prominence=prominence)
        # pl = find_peaks(slice_1,height=0.1,threshold=[0.000000002, 1000],prominence=[0.000000001,1000])

        # pl = find_peaks(slice_1,height=0.1, threshold=[0.2, 100], prominence=[0.05,100])
        pl = pl[0]
        peak_pos = xAxppm[pl]
        cols = [str(round(items,2)) for items in peak_pos]
    else:
        # pl=list(np.where(xAxppm<=peak_pos)[0][0])
        pl=[np.where(xAxppm<=i)[0][0] for i in peak_pos]
        cols = [str(round(items,2)) for items in peak_pos]
        # cols = ['peak']

    slice_1_pl = real_spectrum[0,pl]
    peak_slices = real_spectrum[:,pl]

    fig,ax=plt.subplots()

    # All Slices
    peak_ints=[]
    for slices in real_spectrum:
        current_slice = slices
        peak_ints_now = [float(current_slice[i]) for i in pl]
        peak_ints.append(peak_ints_now)
        plt.plot(xAxppm,current_slice)
        
    # ax.vlines(x=peak_pos, ymin=-0.075, ymax=0.0, color='r')
    ax.vlines(x=peak_pos, ymin=min_slice_1-0.1*max_slice_1, ymax=min_slice_1-0.05*max_slice_1, color='r')
    ax.invert_xaxis()
    ax.set_xlabel("Shift / ppm")
    ax.set_ylabel("Intensity")

    if plottype == 'none':
        plt.close()

    max_peak_ints = np.amax(peak_ints,axis=0)
    peak_ints_norm = []
    for slices2 in peak_ints:
        current_slice2 = np.divide(slices2,max_peak_ints)
        peak_ints_norm.append(current_slice2)

    # peak_ints_norm = peak_ints_norm-peak_ints_norm[-1]
    peak_ints_norm = np.clip(peak_ints_norm,0,1)
    peak_intensity = pd.DataFrame(np.array(peak_ints_norm),columns=cols)
    # display(peak_intensity)

    if norm:
        peak_ints = peak_ints_norm
    
    return peak_ints, cols    

# Plot diffusion data
def diff_plot(peak_ints, datapath, NUC, TD2, plottype='exp', norm=False):
    delta, DELTA, expD, G, gamma = diff_params_import(datapath, NUC, TD2)
    B = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in G]

    if norm:
        ytext="Normalized Intensity"
    else:
        ytext="Intensity"

    fig2,ax2 = plt.subplots()
    if plottype == "exp":
        lines = plt.plot(G,peak_ints, 'o')
        ax2.set_xlabel("Gradient Strength / T m$\mathregular{^{-1}}$")
        ax2.set_ylabel(ytext)
    elif plottype == 'log':
        lines = plt.plot(B,np.log(peak_ints), 'o')#, c='red', mfc='blue', mec='blue')
        ax2.set_xlabel("B / $\mathregular{s m^{-2}}$")
        ax2.set_ylabel(ytext)
    elif plottype == "none":
        plt.close()
    else:
        lines = plt.plot(G,peak_ints, 'o')
        ax2.set_xlabel("Gradient Strength / T m$\mathregular{^{-1}}$")
        ax2.set_ylabel(ytext)
    
    
    grad_params = {"delta": delta, "DELTA": DELTA, "gamma": gamma, "expD":expD}
    return G, grad_params

# Get gamma value from nuclide
def find_gamma(isotope):
    gammalist = [
        ['Name', 'Nuclide', 'Spin', 'Magnetic Moment', 'Gyromagnetic Ratio (MHz/T)', 'Quadrupole Moment (fm^2)'],
        ['Hydrogen', '1H', '0.5', '4.83735', '42.57746460132430', '---'],
        ['Deuterium', '2H', '1', '1.21260', '6.53590463949470', '0.28600'],
        ['Helium', '3He', '0.5', '-3.68515', '-32.43603205003720', '---'],
        ['Tritium', '3H', '0.5', '5.15971', '45.41483118028370', '---'],
        ['Lithium', '6Li', '1', '1.16256', '6.26620067293118', '-0.08080'],
        ['Lithium', '7Li', '1.5', '4.20408', '16.54845000000000', '-4.01000'],
        ['Beryllium', '9Be', '1.5', '-1.52014', '-5.98370064894306', '5.28800'],
        ['Boron', '10B', '3', '2.07921', '4.57519531807410', '8.45900'],
        ['Boron', '11B', '1.5', '3.47103', '13.66297000000000', '4.05900'],
        ['Carbon', '13C', '0.5', '1.21661', '10.70839020506340', '---'],
        ['Nitrogen', '14N', '1', '0.57100', '3.07770645852245', '2.04400'],
        ['Nitrogen', '15N', '0.5', '-0.49050', '-4.31726881729937', '---'],
        ['Oxygen', '17O', '2.5', '-2.24077', '-5.77426865932844', '-2.55800'],
        ['Fluorine', '19F', '0.5', '4.55333', '40.07757016369700', '---'],
        ['Neon', '21Ne', '1.5', '-0.85438', '-3.36307127148622', '10.15500'],
        ['Sodium', '23Na', '1.5', '2.86298', '11.26952278792250', '10.40000'],
        ['Magnesium', '25Mg', '2.5', '-1.01220', '-2.60834261585015', '19.94000'],
        ['Aluminum', '27Al', '2.5', '4.30869', '11.10307854843700', '14.66000'],
        ['Silicon', '29Si', '0.5', '-0.96179', '-8.46545000000000', '---'],
        ['Phosphorus', '31P', '0.5', '1.95999', '17.25144000000000', '---'],
        ['Sulfur', '33S', '1.5', '0.83117', '3.27171633415147', '-6.78000'],
        ['Chlorine', '35Cl', '1.5', '1.06103', '4.17654000000000', '-8.16500'],
        ['Chlorine', '37Cl', '1.5', '0.88320', '3.47653283041643', '-6.43500'],
        ['Potassium', '39K', '1.5', '0.50543', '1.98953228161455', '5.85000'],
        ['Potassium', '40K', '4', '-1.45132', '-2.47372936498302', '-7.30000'],
        ['Potassium', '41K', '1.5', '0.27740', '1.09191431807057', '7.11000'],
        ['Calcium', '43Ca', '3.5', '-1.49407', '-2.86967503240704', '-4.08000'],
        ['Scandium', '45Sc', '3.5', '5.39335', '10.35908000000000', '-22.00000'],
        ['Titanium', '47Ti', '2.5', '-0.93294', '-2.40404000000000', '30.20000'],
        ['Titanium', '49Ti', '3.5', '-1.25201', '-2.40475161264699', '24.70000'],
        ['Vanadium', '50V', '6', '3.61376', '4.25047148768370', '21.00000'],
        ['Vanadium', '51V', '3.5', '5.83808', '11.21327743103380', '-5.20000'],
        ['Chromium', '53Cr', '1.5', '-0.61263', '-2.41152000000000', '-15.00000'],
        ['Manganese', '55Mn', '2.5', '4.10424', '10.57624385581420', '33.00000'],
        ['Iron', '57Fe', '0.5', '0.15696', '1.38156039900351', '---'],
        ['Cobalt', '59Co', '3.5', '5.24700', '10.07769000000000', '42.00000'],
        ['Nickel', '61Ni', '1.5', '-0.96827', '-3.81144000000000', '16.20000'],
        ['Copper', '63Cu', '1.5', '2.87549', '11.31876532731510', '-22.00000'],
        ['Copper', '65Cu', '1.5', '3.07465', '12.10269891500850', '-20.40000'],
        ['Zinc', '67Zn', '2.5', '1.03556', '2.66853501532750', '15.00000'],
        ['Gallium', '69Ga', '1.5', '2.60340', '10.24776396876680', '17.10000'],
        ['Gallium', '71Ga', '1.5', '3.30787', '13.02073645775120', '10.70000'],
        ['Germanium', '73Ge', '4.5', '-0.97229', '-1.48973801382307', '-19.60000'],
        ['Arsenic', '75As', '1.5', '1.85835', '7.31501583241246', '31.40000'],
        ['Selenium', '77Se', '0.5', '0.92678', '8.15731153773769', '---'],
        ['Bromine', '79Br', '1.5', '2.71935', '10.70415668357710', '31.30000'],
        ['Bromine', '81Br', '1.5', '2.93128', '11.53838323328760', '26.20000'],
        ['Krypton', '83Kr', '4.5', '-1.07311', '-1.64423000000000', '25.90000'],
        ['Rubidium', '85Rb', '2.5', '1.60131', '4.12642612503788', '27.60000'],
        ['Strontium', '87Sr', '4.5', '-1.20902', '-1.85246804462381', '33.50000'],
        ['Rubidium', '87Rb', '1.5', '3.55258', '13.98399000000000', '13.35000'],
        ['Yttrium', '89Y', '0.5', '-0.23801', '-2.09492468493000', '---'],
        ['Zirconium', '91Zr', '2.5', '-1.54246', '-3.97478329525992', '-17.60000'],
        ['Niobium', '93Nb', '4.5', '6.82170', '10.45234000000000', '-32.00000'],
        ['Molybdenium', '95Mo', '2.5', '-1.08200', '-2.78680000000000', '-2.20000'],
        ['Molybdenium', '97Mo', '2.5', '-1.10500', '-2.84569000000000', '25.50000'],
        ['Ruthenium', '99Ru', '2.5', '-0.75880', '-1.95601000000000', '7.90000'],
        ['Technetium', '99Tc', '4.5', '6.28100', '9.62251000000000', '-12.90000'],
        ['Ruthenium', '101Ru', '2.5', '-0.85050', '-2.19156000000000', '45.70000'],
        ['Rhodium', '103Rh', '0.5', '-0.15310', '-1.34772000000000', '---'],
        ['Palladium', '105Pd', '2.5', '-0.76000', '-1.95761000000000', '66.00000'],
        ['Silver', '107Ag', '0.5', '-0.19690', '-1.73307000631627', '---'],
        ['Silver', '109Ag', '0.5', '-0.22636', '-1.99239707059020', '---'],
        ['Cadmium', '111Cd', '0.5', '-1.03037', '-9.06914203769978', '---'],
        ['Indium', '113In', '4.5', '6.11240', '9.36547000000000', '79.90000'],
        ['Cadmium', '113Cd', '0.5', '-1.07786', '-9.48709883375341', '---'],
        ['Indium', '115In', '4.5', '6.12560', '9.38569000000000', '81.00000'],
        ['Tin', '115Sn', '0.5', '-1.59150', '-14.00770000000000', '---'],
        ['Tin', '117Sn', '0.5', '-1.73385', '-15.26103326770140', '---'],
        ['Tin', '119Sn', '0.5', '-1.81394', '-15.96595000000000', '---'],
        ['Antimony', '121Sb', '2.5', '3.97960', '10.25515000000000', '-36.00000'],
        ['Antimony', '123Sb', '3.5', '2.89120', '5.55323000000000', '-49.00000'],
        ['Tellurium', '123Te', '0.5', '-1.27643', '-11.23491000000000', '---'],
        ['Tellurium', '125Te', '0.5', '-1.53894', '-13.54542255864230', '---'],
        ['Iodine', '127I', '2.5', '3.32871', '8.57776706639786', '-71.00000'],
        ['Xenon', '129Xe', '0.5', '-1.34749', '-11.86039000000000', '---'],
        ['Xenon', '131Xe', '1.5', '0.89319', '3.51586001685444', '-11.40000'],
        ['Cesium', '133Cs', '3.5', '2.92774', '5.62334202679439', '-0.34300'],
        ['Barium', '135Ba', '1.5', '1.08178', '4.25819000000000', '16.00000'],
        ['Barium', '137Ba', '1.5', '1.21013', '4.76342786926888', '24.50000'],
        ['Lanthanum', '138La', '5', '4.06809', '5.66152329764214', '45.00000'],
        ['Lanthanum', '139La', '3.5', '3.15568', '6.06114544425158', '20.00000'],
        ['Praseodymium', '141Pr', '2.5', '5.05870', '13.03590000000000', '-5.89000'],
        ['Neodymium', '143Nd', '3.5', '-1.20800', '-2.31889000000000', '-63.00000'],
        ['Neodymium', '145Nd', '3.5', '-0.74400', '-1.42921000000000', '-33.00000'],
        ['Samarium', '147Sm', '3.5', '-0.92390', '-1.77458000000000', '-25.90000'],
        ['Samarium', '149Sm', '3.5', '-0.76160', '-1.46295000000000', '7.40000'],
        ['Europium', '151Eu', '2.5', '4.10780', '10.58540000000000', '90.30000'],
        ['Europium', '153Eu', '2.5', '1.81390', '4.67422000000000', '241.20000'],
        ['Gadolinium', '155Gd', '1.5', '-0.33208', '-1.30717137860235', '127.00000'],
        ['Gadolinium', '157Gd', '1.5', '-0.43540', '-1.71394000000000', '135.00000'],
        ['Terbium', '159Tb', '1.5', '2.60000', '10.23525000000000', '143.20000'],
        ['Dysprosium', '161Dy', '0.5', '-0.56830', '-1.46438000000000', '250.70000'],
        ['Dysprosium', '163Dy', '1', '0.79580', '2.05151000000000', '264.80000'],
        ['Holmium', '165Ho', '0.5', '4.73200', '9.08775000000000', '358.00000'],
        ['Erbium', '167Er', '0.5', '-0.63935', '-1.22799179441414', '356.50000'],
        ['Thulium', '169Tm', '1', '-0.40110', '-3.53006000000000', '---'],
        ['Ytterbium', '171Yb', '1.5', '0.85506', '7.52612000000000', '---'],
        ['Ytterbium', '173Yb', '1.5', '-0.80446', '-2.07299000000000', '280.00000'],
        ['Lutetium', '175Lu', '3', '2.53160', '4.86250000000000', '349.00000'],
        ['Lutetium', '176Lu', '1.5', '3.38800', '3.45112000000000', '497.00000'],
        ['Hafnium', '177Hf', '0.5', '0.89970', '1.72842000000000', '336.50000'],
        ['Hafnium', '179Hf', '1', '-0.70850', '-1.08560000000000', '379.30000'],
        ['Tantalum', '181Ta', '0.5', '2.68790', '5.16267000000000', '317.00000'],
        ['Tungsten', '183W', '2.5', '0.20401', '1.79564972994000', '---'],
        ['Rhenium', '185Re', '0.5', '3.77100', '9.71752000000000', '218.00000'],
        ['Osmium', '187Os', '1.5', '0.11198', '0.98563064707380', '---'],
        ['Rhenium', '187Re', '1.5', '3.80960', '9.81700000000000', '207.00000'],
        ['Osmium', '189Os', '2.5', '0.85197', '3.35360155237225', '85.60000'],
        ['Iridium', '191Ir', '2.5', '0.19460', '0.76585000000000', '81.60000'],
        ['Iridium', '193Ir', '0.5', '0.21130', '0.83190000000000', '75.10000'],
        ['Platinum', '195Pt', '0.5', '1.05570', '9.29226000000000', '---'],
        ['Gold', '197Au', '1.5', '0.19127', '0.75289837379052', '54.70000'],
        ['Mercury', '199Hg', '1.5', '0.87622', '7.71231431685275', '---'],
        ['Mercury', '201Hg', '1.5', '-0.72325', '-2.84691587554490', '38.60000'],
        ['Thallium', '203Tl', '1.5', '2.80983', '24.73161181836180', '---'],
        ['Thallium', '205Tl', '4', '2.83747', '24.97488014887780', '---'],
        ['Lead', '207Pb', '1.5', '1.00906', '8.88157793726598', '---'],
        ['Bismuth', '209Bi', '3.5', '4.54440', '6.96303000000000', '-51.60000'],
        ['Uranium', '235U', '3.5', '-0.43000', '-0.82761000000000', '493.60000']
        ]

    df = pd.DataFrame(gammalist)
    df.columns = df.iloc[0]
    df = df[1:]
    if df['Nuclide'].isin([isotope]).any():
        gamma = df.loc[df['Nuclide'] == isotope,"Gyromagnetic Ratio (MHz/T)"]
        gamma = float(gamma.iloc[0])*1e6
    else:
        print("Isotope string not recognised, please input a string of format e.g., '1H' or '27Al' and ensure it is an NMR active nuclide.")
        return
    

    return gamma

# Head function
def diffusion_read_plot(datapath, procno=1, f2l=5,f2r=0,prominence=0.005, plottype="exp",norm=False, peak_pos=float("NaN")):
    xAxppm, real_spectrum, expt_parameters = xf2(datapath, procno=procno, f2l=f2l, f2r=f2r)
    peak_ints, peaklist = xf2_peak_pick(xAxppm, real_spectrum,prominence=prominence,norm=norm, peak_pos=peak_pos, plottype=plottype)
    if np.shape(peak_ints)[1] > 25:
        print("Too many peaks found, check prominence parameter and processing then retry.")
        switch = 0
        return
    else:
        switch = 1
    if switch:
        G, grad_params = diff_plot(peak_ints, datapath, expt_parameters["NUC"], TD2=expt_parameters["TD_2"],plottype=plottype)
        return G, peak_ints, peaklist, grad_params
    else:
        return

# Gradient biexp fitting function using LMFit
def DiffBiExpLM(G, y, grad_params, expD1 = 1.00001e-10, expD2 = 0, plottype="exp", stretch=False, annotation=True, peaklist = False):
    # Update diffusion function to read B values directly, avoiding issues of using the wrong form of the Stejskal-Tanner equation.
    """
    expD1, expD2 = expD
    DiffBiExpLM(G, y, grad_params, expD1 = grad_params['expD'], expD2 = grad_params['expD'], plottype="exp"):
    units used:
    D [=] m2/s
    delta[=] s (read in from data file in [ms])
    DELTA [=] s (read in from data file in [ms])
    G [=] T/m (read in from data file in [G/cm])
    gamma [=] 1/(T s)
    """

    delta = grad_params['delta']
    DELTA = grad_params['DELTA']
    gamma = grad_params['gamma']
    if expD1 == 1.00001e-10:
        expD1 = grad_params['expD']

    yweight = y
    maxy = max(y)
    y = np.divide(y,maxy)
    # y[y == 0]
    
    # ylog = y[y!=0]
    ylog = np.log(y)
    ylog[ylog == -np.inf] = 'nan'
    yweight[yweight == 0] = 1e-15
    
    # print(y)

    B = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in G]
    Gsmooth = np.linspace(0, G[-1],100)
    Bsmooth = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in Gsmooth]
    # method_choice = 'trf'
    choice = 1
    evalstr = """plt.plot(G, DiffBiExpFit, '--', color='red', label=method_str[choice]+" fit")"""

    def DiffDoubleExp(B, D1, D2, I01, I02, beta):

        # result1 = I01 * np.exp(np.multiply(-D1,B))
        result1 = I01 * np.exp(-1 * np.multiply(D1,B)**beta)
        # result1 = np.multiply(-D1,B)
        # result1 = np.power(result1,beta)
        # result1 = I01 * np.exp(result1)
        result2 = 0
        if expD2 != 0:
            result2 = I02 * np.exp(np.multiply(-D2,B))
        result = result1+result2
        return result
    
    method_str = ["Mono-exponential", "Bi-exponential", "Tri-exponential", "Stretched Exponential"]

    # Double-exponential Fit

    fmodel = Model(DiffDoubleExp)

    # Building model parameters
    params = Parameters()
    params.add("D1", min = expD1/50, max = expD1*50, value = expD1)
    if expD2 != 0:
        params.add("D2", min = expD2/50, max = expD2*50, value = expD2)
    else:
        params.add("D2", value = expD2, vary = False)
    
    if expD2 != 0:
        params.add("I01", min = 0.0, max = 1.5, value = 0.5)
        params.add("I02", min = 0.0, max = 1.5, value = 0.5)
        # params.add("I02", min = 0.0, max = 1.2, expr = "1.0-I01")
    else:
        params.add("I01", min = 0.0, max = 1.5, value = 1.0, vary = True)
        params.add("I02", value = 0.0, vary = False)
        print("Proceeding with one parameter fitting.")

    params.add("beta", min=0.0, max=5.0, value=1.0, vary=stretch)

    # Run Model
    DiffBiExpFit = fmodel.fit(y,params,B=B)
    # DiffBiExpFit = fmodel.fit(y,params,B=B, weights=np.divide(y,1), scale_covar=True)
    y_interp = DiffBiExpFit.model.func(Bsmooth, **DiffBiExpFit.best_values)
    model_fits = DiffBiExpFit.fit_report()
    
    D1 = DiffBiExpFit.best_values["D1"]
    D2 = DiffBiExpFit.best_values["D2"]
    I01 = DiffBiExpFit.best_values["I01"]
    I02 = DiffBiExpFit.best_values["I02"]
    w1 = I01/(I01+I02)
    w2 = I02/(I01+I02)
    beta = DiffBiExpFit.best_values["beta"]
    summary = DiffBiExpFit.summary()
    r2 = summary["rsquared"]
    chisqr = summary["chisqr"]
    delmodel = DiffBiExpFit.eval_uncertainty(sigma=3)

    results = DiffBiExpFit.params
    err_D1 = results['D1'].stderr
    err_D2 = results['D2'].stderr
    err_I01 = results['I01'].stderr
    err_I02 = results['I02'].stderr
    # print(delmodel)
    def Diff_Range(G, B, D1, I01, D2, I02, beta):

        y_p05 = (I01+err_I01) * np.exp(-1 * np.multiply((D1+err_D1),B)**beta)
        y_m05 = (I01-err_I01) * np.exp(-1 * np.multiply((D1-err_D1),B)**beta)

        y_range1 = np.append(y_p05, y_m05[::-1])

        y_range2 = 0
        if expD2 != 0:
            y_p05 = (I02+err_I02) * np.exp(-1 * np.multiply(D2+err_D2,B)**beta)
            y_m05 = (I02+err_I02) * np.exp(-1 * np.multiply(D2-err_D2,B)**beta)
            y_range2 = np.append(y_p05, y_m05[::-1])

        y_range = y_range1+y_range2

        if plottype == "exp" or "exp_range":
            x_range = np.append(G, G[::-1])
        else:
            x_range = np.append(B, B[::-1])
        return x_range, y_range
    # def Diff_Range(G, B, D1, I01, D2, I02, beta):

    #     y_p05 = I01 * np.exp(-1 * np.multiply(D1*1.05,B)**beta)
    #     y_m05 = I01 * np.exp(-1 * np.multiply(D1*0.95,B)**beta)

    #     y_range1 = np.append(y_p05, y_m05[::-1])

    #     y_range2 = 0
    #     if expD2 != 0:
    #         y_p05 = I02 * np.exp(-1 * np.multiply(D2+err_D2,B)**beta)
    #         y_m05 = I02 * np.exp(-1 * np.multiply(D2-err_D2,B)**beta)
    #         y_range2 = np.append(y_p05, y_m05[::-1])

    #     y_range = y_range1+y_range2

    #     if plottype == "exp" or "exp_range":
    #         x_range = np.append(G, G[::-1])
    #     else:
    #         x_range = np.append(B, B[::-1])
    #     return x_range, y_range
    
    x_range, y_range = Diff_Range(Gsmooth, Bsmooth, D1, I01, D2, I02, beta)

    boxprops = dict(facecolor='white', alpha=0.8, linewidth=0)
    if annotation:
        if peaklist==False:
            axtext = "$\mathregular{w_1}$: "+f"{w1: .3f}"+", $\mathregular{w_2}$: "+f"{w2: .3f}"+"\n$\mathregular{D_1}$: "+f"{D1:.3e}"+", $\mathregular{D_2}$: "+f"{D2:.3e}"+"\n$\mathregular{\\beta}$: "+f"{beta:.3f}""\n$\mathregular{R^2}$: "+f"{r2:.3f}"
        else:
            axtext = "$\mathregular{w_1}$: "+f"{w1: .3f}"+", $\mathregular{w_2}$: "+f"{w2: .3f}"+"\n$\mathregular{D_1}$: "+f"{D1:.3e}"+", $\mathregular{D_2}$: "+f"{D2:.3e}"+"\n$\mathregular{\\beta}$: "+f"{beta:.3f}""\n$\mathregular{R^2}$: "+f"{r2:.3f}"+"\n$\mathregular{\\delta}$: "+peaklist
    else:
        axtext = ""
    if plottype == "exp":
        fig, ax = plt.subplots()
        ax.plot(Gsmooth, y_interp, '--', color='red', label="Fit")
        ax.plot(G, y, 'o', color='blue', linestyle='none', label="Experimental Data")
        # ax.errorbar(B, ylog, abs(-np.log(delmodel)), marker = 'o', mfc = 'b',mec = 'b', linestyle = 'none', ecolor='b',capsize=4.0)
        ax.set_xlabel("Gradient Strength / $\mathregular{T m^{-1}}$")
        # ax.set_xlabel("B / $\mathregular{s m^{-2}}$")
        # plt.text(0e11, min(ylog), "$\mathregular{w_1}$: "+str(round(I01,3))+"\n$\mathregular{w_2}$: "+str(round(I02,3)), fontsize = 18)
        ax.text(.95, .95, axtext, fontsize = 16, backgroundcolor='white', 
                bbox=boxprops, ha='right', va='top', transform=ax.transAxes)
        ax.set_ylabel("Intensity")
        ax.set_ylim(-0.05, 1.1)
        fig.set_figheight(5)
        fig.set_figwidth(6)
    elif plottype == "expB":
        fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        # # plt.fill_between(G, y+delmodel, y-delmodel, color='red',alpha=0.3)
        # ax[0].plot(Bsmooth, y_interp, '--', color='red', label="Experimental Data")
        # ax[0].errorbar(B, y, delmodel, marker = 'o', mfc = 'b',mec = 'b', linestyle = 'none', ecolor='b',capsize=4.0)
        # ax[0].plot(B, y, 'o', color='blue', label="Experimental Data")
        # ax[0].set_xlabel("G / T$\mathregular{m^{-1}}$")
        ax[0].text(.95, .95, axtext, fontsize = 16, backgroundcolor='white', 
                   bbox=boxprops, ha='right', va='top', transform=ax[0].transAxes)
        ax[0].plot(Bsmooth, y_interp, '--', color='red', label="Experimental Data")
        ax[0].errorbar(B, y, delmodel, marker = 'o', mfc = 'b',mec = 'b', linestyle = 'none', ecolor='b',capsize=4.0)
        DiffBiExpFit.plot_residuals(ax=ax[1],title=" ", yerr=None)
        ax[0].set_xlabel(None)
        ax[0].set_ylim(-0.05, 1.1)
        ax[1].set_xlabel("B / s$\mathregular{m^{-2}}$")
        ax[1].set_ylim(-0.1, 0.1)
        ax[0].set_ylabel("Intensity")
        fig.set_figheight(8)
        fig.set_figwidth(6)
    elif plottype == "log":
        fig, ax = plt.subplots()
        ax.plot(Bsmooth, np.log(y_interp), '--', color='red', label="Fit")
        ax.plot(B, ylog, 'o', color='blue', linestyle='none', label="Experimental Data")
        # ax.errorbar(B, ylog, abs(-np.log(delmodel)), marker = 'o', mfc = 'b',mec = 'b', linestyle = 'none', ecolor='b',capsize=4.0)
        ax.set_xlabel("B / s$\mathregular{m^{-2}}$")
        # ax.set_xlabel("B / $\mathregular{s m^{-2}}$")
        # plt.text(0e11, min(ylog), "$\mathregular{w_1}$: "+str(round(I01,3))+"\n$\mathregular{w_2}$: "+str(round(I02,3)), fontsize = 18)
        ax.text(.95, .95, axtext, fontsize = 16, backgroundcolor='white', 
                bbox=boxprops, ha='right', va='top', transform=ax.transAxes)
        ax.set_ylabel("Intensity")
        fig.set_figheight(5)
        fig.set_figwidth(6)
    elif plottype == "exp_range":
        fig, ax = plt.subplots()
        ax.plot(Gsmooth, y_interp, '--', color='red', label="Fit")
        ax.plot(G, y, 'o', color='blue', linestyle='none', label="Experimental Data")
        ax.fill(x_range, y_range, '--', color='grey', alpha=0.4, label="uncertainty")
        ax.set_xlabel("Gradient Strength / $\mathregular{T m^{-1}}$")
        ax.text(.95, .95, axtext, fontsize = 16, backgroundcolor='white', 
                bbox=boxprops, ha='right', va='top', transform=ax.transAxes)
        ax.set_ylabel("Intensity")
        ax.set_ylim(-0.05, 1.1)
        fig.set_figheight(5)
        fig.set_figwidth(6)
    elif plottype == "none":
        plt.close()
    else:
        fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax[0].text(.95, .95, axtext, fontsize = 16, backgroundcolor='white', 
        bbox=boxprops, ha='right', va='top', transform=ax[0].transAxes)
        # ax[0].text(.95, .95, "$\mathregular{w_1}$: "+f"{w1: .3f}"+", $\mathregular{w_2}$: "+f"{w2: .3f}"+
        #     "\n$\mathregular{D_1}$: "+f"{D1:.3e}"+", $\mathregular{D_2}$: "+f"{D2:.3e}"+
        #     "\n$\mathregular{R^2}$: "+f"{r2:.3f}", 
        #     fontsize = 16, backgroundcolor='white', bbox=boxprops, ha='right', va='top', transform=ax[0].transAxes)
        ax[0].plot(Bsmooth, y_interp, '--', color='red', label="Experimental Data")
        ax[0].errorbar(B, y, delmodel, marker = 'o', mfc = 'b',mec = 'b', linestyle = 'none', ecolor='b',capsize=4.0)
        DiffBiExpFit.plot_residuals(ax=ax[1],title=" ", yerr=None)
        # ax[1].plot(B,residuals, 'bo')
        # err_dict = {capsize: 4.0}
        # DiffBiExpFit.plot_residuals(ax=ax[1],title=" ",datafmt='bo')#,fit_kws={"capsize":4.0})
        ax[0].set_xlabel(None)
        ax[1].set_xlabel("B / s$\mathregular{m^{-2}}$")
        ax[1].set_ylim(-0.1, 0.1)
        ax[0].set_ylabel("Intensity")
        fig.set_figheight(8)
        fig.set_figwidth(6)
    

    fit_results = {"D1": D1, "I01": w1, "D2": D2, "I02": w2, "D1 Error": err_D1, "D2 Error": err_D2, "I01 Error": err_I01, "I02 Error": err_I02, "beta": beta, "G":G, "B":B, "y":y, "G_fit":Gsmooth, "B_fit":Bsmooth, "y_fit":y_interp}
    return fit_results

# Gradient fitting function - as linear
def Diffusion_Fit(delta, DELTA, G, y, gamma, p0=1e-10,showall=False,fittype = "default"):
    """
    p0 = expD
    Diffusion_Fit(G, y,p0=0.0000000001,showall=False,fittype = "default")
    units used:
    D [=] m2/s
    delta[=] s (read in from data file in [ms])
    DELTA [=] s (read in from data file in [ms])
    G [=] T/m (read in from data file in [G/cm])
    gamma [=] 1/(T s)
    """
    y = np.clip(y, 0, 1)

    B = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in G]
    method_choice = 'trf'
    if fittype == "default":
        choice = 0
        evalstr = """plt.plot(G, np.exp(DiffMonoExpFit), '--', color='red', label=method_str[choice]+" fit")"""
        # np.exp(DiffMonoExpFit)
        # np.exp(DiffBiExpFit)
        # np.exp(DiffStretchExpFit)
    elif fittype == "Mono-exponential":
        choice = 0
        evalstr = """plt.plot(G, np.exp(DiffMonoExpFit), '--', color='red', label=method_str[choice]+" fit")"""
    elif fittype == "Bi-exponential":
        choice = 1
        evalstr = """plt.plot(G, np.exp(DiffBiExpFit), '--', color='red', label=method_str[choice]+" fit")"""
    elif fittype == "Stretched exponential":
        choice = 3
        evalstr = """plt.plot(G, np.exp(DiffStretchExpFit), '--', color='red', label=method_str[choice]+" fit")"""
    else:
        choice = 0
        evalstr = """plt.plot(G, np.exp(DiffMonoExpFit), '--', color='red', label=method_str[choice]+" fit")"""

    def DiffMonoExp(B, D):
        # result = [-1 * (D * (2*np.pi*gamma*delta*i)**2) * ((DELTA)-((delta)/3)) for i in G]
        # result = [(-D * i) for i in B]
        result = np.multiply(-D,B)
        # result=[]
        # for i in G:
        #     result.append(np.exp(-D * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2))
            # print(result)
        return result
    
    # def DiffMonoExp(G, D):
    #     result = [np.exp(-1 * (D * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2)) for i in G]
    #     # result=[]
    #     # for i in G:
    #     #     result.append(np.exp(-D * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2))
    #         # print(result)
    #     return result

    def DiffDoubleExp(B, D1, D2, I01, I02):
        # result=[]
        # result = [np.log(I02) - (D1 * (2*np.pi*gamma*delta*i)**2 * ((DELTA)-((delta)/3))) + np.log(1-I02) - (D2 * (2*np.pi*gamma*delta*i)**2 * ((DELTA)-((delta)/3))) for i in G]
        # result = [(np.log(I02) - (D1 * ((2*np.pi*gamma*delta*i)**2 * ((DELTA)-((delta)/3))))) + (np.log(1-I02) - (D2 * ((2*np.pi*gamma*delta*i)**2 * ((DELTA)-((delta)/3)))))for i in G]
        # result = [np.log(I02) + (-D1 * i) + np.log(1-I02) + (-D2 * i) for i in B]

        result1 = np.log(I01) + np.multiply(-D1,B)
        result2 = np.log(I02) + np.multiply(-D2,B)
        result = result1+result2
        # for i in G:
        #     result = [(I01 * np.exp(-1 * (D1 * (2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3))))) + (I02 * np.exp(-1 * (D2 * (2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3))))) for i in G]
        return result
    
    # def DiffTripleExp(G, D1, D2, D3, I01, I02, I03):
    #     result=[]
    #     for i in G:
    #         result.append(\
    #         I01 * np.exp(-1* (D1 * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2)) + \
    #         I02 * np.exp(-1 * (D2 * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2)) + \
    #         I03 * np.exp(-1 * (D3 * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2)))
    #     return result
    
    def DiffStretchExp(B, D, beta, I_str):
        result = I_str + np.multiply(-D,B)**1
        # result = [np.log(I_str) + (-D * i)**beta for i in B]
        # result = [np.log(I_str) - ((D * B)**beta) for i in B]
        # result = [np.exp(((-D * (2*np.pi*gamma)**2 * (delta**2) * ( DELTA-(delta/3)) * (i**2)))**beta) for i in G]
        # result=[]
        # for i in G:
        #     result.append(np.exp((-D * (2*np.pi*gamma)**2 * ((delta)**2) * ((DELTA)-((delta)/3)) * (i)**2)**beta))
        return result
    
    def custom_weighting_function(x_data, a, b, max_weight=1.0, min_weight=0.1):
        weights = np.ones_like(x_data)  # Initialize with equal weights
        mask = (x_data >= a) & (x_data <= b)  # Define a mask for the undesired range
        weights[mask] = min_weight  # Set a lower weight for data points in the range
        weights[~mask] = max_weight  # Set a higher weight for data points outside the range
        return weights

    method_str = ["Mono-exponential", "Bi-exponential", "Tri-exponential", "Stretched Exponential"]

    # Mono-exponential Fit
    param_bounds1 = (1e-15, 1e-6)
    params, cv = scipy.optimize.curve_fit(DiffMonoExp, B, np.log(y), p0, bounds = param_bounds1, method = method_choice)
    D = params
    print(f"Mono-exponential Fit: \nD = {D} m^2/s")
    DiffMonoExpFit = DiffMonoExp(B,D)
    R_sq_MonoExp = r2_score(np.log(y), DiffMonoExpFit)

    # Double-exponential Fit
    
    bip0 = (p0*10, p0, 0.5,0.5)
    param_bounds2 = ([1e-15, 1e-15, 0.0001, 0.0001], [1e-6, 1e-6, 100, 100])
    range_bottom = 0.5
    range_top = 1.0
    params, cv = scipy.optimize.curve_fit(DiffDoubleExp, B, np.log(y), bip0, bounds = param_bounds2, maxfev = 100000, method = method_choice)#, sigma=custom_weighting_function(y, range_bottom, range_top))
    D1, D2, I01, I02 = params 
    w1 = I01
    w2 = I02
    print(f"Bi-exponential Fit: \nD1 = {D1} m^2/s, w1 = {w1}\nD2 = {D2} m^2/s, w2 = {w2}")
    DiffBiExpFit = DiffDoubleExp(B,D1,D2,I01,I02)
    DiffMonoExpFitforBi1 = DiffMonoExp(B,D1)
    DiffMonoExpFitforBi2 = DiffMonoExp(B,D2)
    R_sq_BiExp = r2_score(np.log(y), DiffBiExpFit)

    # Stretched Exponential Fit
    beta_guess = 1
    I_guess = 0.5
    guesses = (p0, beta_guess, I_guess)
    param_bounds_str = ([1e-15, 0, 0], [1e-6, 1, 10])
    params, cv = scipy.optimize.curve_fit(DiffStretchExp, B, np.log(y), guesses, bounds = param_bounds_str, method = method_choice)
    D_str, beta, I_str = params
    print(f"Stretched Exponential Fit: \nD = {D_str} m^2/s \n\u03B2 = {beta} \nA = {I_guess}")
    DiffStretchExpFit = DiffStretchExp(B,D_str,beta,I_str)
    R_sq_Stretch = r2_score(np.log(y), DiffStretchExpFit)
    print(R_sq_MonoExp, R_sq_BiExp, R_sq_Stretch)

    plt.plot(G, y, 'o', color='black', label="Experimental Data")
    plt.plot(G, np.exp(DiffMonoExpFit), '--', color='teal', linewidth = 2, label=method_str[0]+" fit")
    plt.plot(G, np.exp(DiffBiExpFit), '-.', color='orange', label=method_str[1]+" fit")
    plt.plot(G, np.exp(DiffStretchExpFit), ':', color='red', label=method_str[3]+" fit")
    plt.xlabel('Gradient Strength / $\mathregular{T m^{–1}}$')
    plt.ylabel('Normalized intensity')
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(B, np.log(y), 'o', color='black', label="Experimental Data")
    plt.plot(B, DiffMonoExpFit, '--', color='teal', linewidth = 2, label=method_str[0]+" fit")
    plt.plot(B, DiffBiExpFit, '-.', color='orange', label=method_str[1]+" fit")
    plt.plot(B, DiffStretchExpFit, ':', color='red', label=method_str[3]+" fit")
    plt.plot(B, DiffMonoExpFitforBi1, ':', color='blue', label="bi fit-1")
    plt.plot(B, DiffMonoExpFitforBi2, ':', color='green', label="bi fit-2")
    plt.xlabel('B / $\mathregular{s m^{–2}}$')
    plt.ylabel('Normalized intensity')
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(G, y, 'o', color='blue', label="Experimental Data")
    eval(evalstr)
    # plt.plot(G, np.exp(DiffStretchExpFit), '--', color='red', label=method_str[choice]+" fit")
    plt.xlabel('Gradient Strength / $\mathregular{T m^{–1}}$')
    plt.ylabel('Normalized intensity')
    plt.legend(loc="upper right")
    plt.show()


def sim_diffusion(NUC, delta = 1, DELTA = 20, maxgrad = 17, D = 0):
    """
    fig, ax = sim_diffusion(NUC, delta=1, DELTA = 20, maxgrad = 17, D = 0)

    Function to help estimate appropriate diffusion experiment parameters. Can set the maximum gradient, 
    little delta, and big DELTA to understand the level of attenuation/shape of the curve for whatever nuclide.

    D = 0 is a placeholder, if left as 0, D will be a range of 1e-7 to 1e-15 stepping by order of magnitude,
    if a value for D is set, only one line will be plotted.
    """

    from matplotlib import cm

    switch = 1
    delta = delta/1000
    DELTA = DELTA/1000
    if D == 0:
        D = np.logspace(-8,-15,8)
        switch = 0
    gamma = find_gamma(NUC)  # [10^7 1/T/s]
    G = np.arange(0,maxgrad+(maxgrad/100.0),maxgrad/99.0)
    B = [(2*np.pi*gamma*delta*i)**2 * (DELTA-(delta/3)) for i in G]

    if switch == 0:
        I = np.zeros(shape=(len(D),len(G)))
        cnt=0
        for j in D:
            Inow = np.exp(np.multiply(-j,B))
            I[cnt] = Inow
            cnt+=1
    else:
        I = np.exp(np.multiply(-D,B))
    

    fig, ax = plt.subplots()
    if switch == 0:
        colmap=cm.seismic(np.linspace(0,1,len(D)))
        [plt.plot(G,I[k,:],color=c, linewidth=2,label=str(D[k])+" $\mathregular{m^2 s^{–1}}$") for k,c in zip(range(len(D)),colmap)]
    else:
        plt.plot(G,I,linewidth=2,color='r',label=str(D)+" $\mathregular{m^2 s^{–1}}$")
    ax.set_xlim(0,maxgrad*1.25)
    plt.legend(loc='upper right',frameon=False)
    plt.xlabel('Gradient Strength, g / $\mathregular{T m^{–1}}$')
    plt.ylabel("Intensity, $\mathregular{I/I_0}$")
    plt.show()
    return fig,ax

def eNMR_xf2(datapath, procno=1):
    """xAxppm, real_spectrum, imaginary_spectrum, expt_parameters = eNMR_xf2(datapath, procno=1)
    expt_parameters = {'NUC': NUC, "SF": SF, "OBS": OBS, "x_Hz": xAxHz, "TD": TD, "TD_2":TD_2, "voltage_list": v_list}
    """
    real_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2rr")
    imaginary_spectrum_path = os.path.join(datapath,"pdata",str(procno),"2ii")
    imaginary_spectrum_path_ir = os.path.join(datapath,"pdata",str(procno),"2ir")

    procs = os.path.join(datapath,"pdata",str(procno),"procs")
    acqus = os.path.join(datapath,"acqus")
    proc2s = os.path.join(datapath,"pdata",str(procno),"proc2s")
    acqu2s = os.path.join(datapath,"acqu2s")

    ########################################################################

    # Bruker file format information
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Bruker binary files (ser/fid) store data as an array of numbers whose
    # endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
    # endian), and whose data type is determined by the parameter DTYPA (0 = int32,
    # 2 = float64). Typically the direct dimension is digitally filtered. The exact
    # method of removing this filter is unknown but an approximation is available.
    # Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
    # `JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
    # prefixed with a '$'.

    ####################################

    # Get aqcus
    O1str = '##$O1= '
    OBSstr = '##$BF1= '
    NUCstr = '##$NUC1= <'
    Lstr = "##$L= (0..31)"
    CNSTstr = "##$CNST= (0..63)"
    TDstr = "##$TD= "

    O1 = float("NaN")
    OBS = float("NaN")
    NUC = ""
    L1 = float("NaN")
    L2 = float("NaN")
    CNST31 = float("NaN")
    TD = float("NaN")

    with open(acqus,"rb") as input:
        for line in input:
            if O1str in line.decode():
                linestr = line.decode()
                O1 = float(linestr[len(O1str):len(linestr)-1])
            if OBSstr in line.decode():
                linestr = line.decode()
                OBS = float(linestr[len(OBSstr):len(linestr)-1])
            if NUCstr in line.decode():
                linestr = line.decode()
                NUC = str(linestr[len(NUCstr):len(linestr)-2])
            if TDstr in line.decode():
                linestr = line.decode()
                TD = float(linestr.strip(TDstr))
            if Lstr in line.decode():
                line = next(input)
                linestr = line.decode()
                L = (linestr.strip("\n").split(" "))
                L1 = float(L[1])
                L2 = float(L[2])
            if CNSTstr in line.decode():
                CNST = []
                line = next(input)
                while "##$CPDPRG=" not in str(line):
                    linestr = line.decode()
                    CNST.extend(linestr.strip("\n").split(" "))
                    line = next(input)
                CNST31 = float(CNST[31])
            if ~np.isnan(O1) and ~np.isnan(OBS) and ~np.isnan(L1) and ~np.isnan(TD) and ~np.isnan(CNST31) and not len(NUC)==0:
                break

    ####################################

    # Get procs

    SWstr = '##$SW_p= '
    SIstr = '##$SI= '
    SFstr = '##$SF= '
    NCstr = '##$NC_proc= '
    XDIM_F2str = '##$XDIM= '

    SW = float("NaN")
    SI = int(0)
    SF = float("NaN")
    NC_proc = float("NaN")
    XDIM_F2 = int(0)


    with open(procs,"rb") as input:
        for line in input:
            if SWstr in line.decode():
                linestr = line.decode()
                SW = float(linestr[len(SWstr):len(linestr)-1])
            if SIstr in line.decode():
                linestr = line.decode()
                SI = int(linestr[len(SIstr):len(linestr)-1])
            if SFstr in line.decode():
                linestr = line.decode()
                SF = float(linestr[len(SFstr):len(linestr)-1])
            if NCstr in line.decode():
                linestr = line.decode()
                NC_proc = float(linestr[len(NCstr):len(linestr)-1])
            if XDIM_F2str in line.decode():
                linestr = line.decode()
                XDIM_F2 = int(linestr[len(XDIM_F2str):len(linestr)-1])
            if ~np.isnan(SW) and SI!=int(0) and ~np.isnan(NC_proc) and ~np.isnan(SF) and XDIM_F2!=int(0):
                break

    ####################################

    # Get aqcu2s for indirect dimension
    O1str_2 = '##$O1= '
    OBSstr_2 = '##$BF1= '
    NUCstr_2 = '##$NUC1= <'
    TDstr_2 = "##$TD= "

    O1_2 = float("NaN")
    OBS_2 = float("NaN")
    NUC_2 = ""
    TD_2 = float("NaN")

    with open(acqu2s,"rb") as input:
        for line in input:
    #         print(line.decode())
            if O1str_2 in line.decode():
                linestr = line.decode()
                O1_2 = float(linestr[len(O1str_2):len(linestr)-1])
            if OBSstr_2 in line.decode():
                linestr = line.decode()
                OBS_2 = float(linestr[len(OBSstr_2):len(linestr)-1])
            if NUCstr_2 in line.decode():
                linestr = line.decode()
                NUC_2 = str(linestr[len(NUCstr_2):len(linestr)-2])
            if TDstr_2 in line.decode():
                linestr = line.decode()
                TD_2  = float(linestr.strip(TDstr_2))
            if ~np.isnan(O1_2) and ~np.isnan(OBS_2) and ~np.isnan(TD_2) and not len(NUC_2)==0:
                break

    ####################################

    # # Get proc2s for indirect dimension

    SIstr_2 = '##$SI= '
    XDIM_F1str = '##$XDIM= '

    XDIM_F1 = int(0)
    SI_2 = int(0)

    with open(proc2s,"rb") as input:
        for line in input:
            if SIstr_2 in line.decode():
                linestr = line.decode()
                SI_2 = int(linestr[len(SIstr_2):len(linestr)-1])
            if XDIM_F1str in line.decode():
                linestr = line.decode()
                XDIM_F1 = int(linestr[len(XDIM_F1str):len(linestr)-1])
            if SI_2!=int(0) and XDIM_F1!=int(0):
                break

    ####################################

    # Determine x axis values
    SR = (SF-OBS)*1000000
    true_centre = O1-SR
    xmin = true_centre-SW/2
    xmax = true_centre+SW/2
    xAxHz = np.linspace(xmax,xmin,num=int(SI))
    xAxppm = xAxHz/SF

    real_spectrum = np.fromfile(real_spectrum_path, dtype='<i4', count=-1)
    try:
        imaginary_spectrum = np.fromfile(imaginary_spectrum_path, dtype='<i4', count=-1)
    except:
        imaginary_spectrum = np.fromfile(imaginary_spectrum_path_ir, dtype='<i4', count=-1)
    if not bool(real_spectrum.any()) or not bool(imaginary_spectrum.any()):
        print("Error: Spectrum not read.")
        
    if XDIM_F1 == 1:
        real_spectrum = real_spectrum.reshape([int(SI_2),int(SI)])

        imaginary_spectrum = imaginary_spectrum.reshape([int(SI_2),int(SI)])
    else:
        # to shape the column matrix according to Bruker's format, matrices are broken into (XDIM_F1,XDIM_F2) submatrices, so reshaping where XDIM_F1!=1 requires this procedure.
        column_matrix_real = real_spectrum
        column_matrix_imag = imaginary_spectrum
        submatrix_rows = int(SI_2 // XDIM_F1)
        submatrix_cols = int(SI // XDIM_F2)
        submatrix_number = submatrix_cols*submatrix_rows

        blocks_real = np.array(np.array_split(column_matrix_real,submatrix_number))  # Split into submatrices
        blocks_real = np.reshape(blocks_real,(submatrix_rows,submatrix_cols,-1)) # Reshape these submatrices so each has its own 1D array

        blocks_imag = np.array(np.array_split(column_matrix_imag,submatrix_number))  # Split into submatrices
        blocks_imag = np.reshape(blocks_imag,(submatrix_rows,submatrix_cols,-1)) # Reshape these submatrices so each has its own 1D array

        real_spectrum = np.vstack([np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks_real])  # Concatenate submatrices in the correct orientation
        imaginary_spectrum = np.vstack([np.hstack([np.reshape(c, (XDIM_F1, XDIM_F2)) for c in b]) for b in blocks_imag])  # Concatenate submatrices in the correct orientation

    real_spectrum = real_spectrum*2**NC_proc
    imaginary_spectrum = imaginary_spectrum*2**NC_proc

    expt_parameters = {'NUC': NUC, "TD_2": TD_2}
    # get vd list
    vd_list_datapath = os.path.join(datapath,"vdlist")
    vd = pd.read_csv(vd_list_datapath,header=None, names=['Delay'])
    vd['Delay'] = pd.to_numeric(vd['Delay'].str.replace('m', 'e-3').str.replace('u', 'e-6'))
    v_list = [(delay-0.5e-6)*2e6 for delay in vd['Delay']] # convert delay to voltages
    v_list = [item if index % 2 == 0 else item * -1 for index, item in enumerate(v_list)]
    
    expt_parameters = {'NUC': NUC, "SF": SF, "OBS": OBS, "x_Hz": xAxHz, "TD": TD, "TD_2":TD_2, "voltage_list": v_list}

    return xAxppm, real_spectrum, imaginary_spectrum, expt_parameters

def eNMR(xax, real_spec, com_spec, exp_params, f1p=0, f2p=0, gaussian_component = False, prominence=0.25, S0_init = 1, R_init = 0.1, component='complex'):

    f2l_temp = max(xax)
    f2r_temp = min(xax)

    if f1p>f2p:
        f1p, f2p = f2p,f1p
    elif f1p==f2p==0:
        f1p = np.argmax(xax==f2l_temp)
        f2p = np.argmax(xax==f2r_temp)

    xlow = np.argmax(xax<f1p)
    xhigh = np.argmax(xax<f2p)

    if f2p == 0:
        xlow = np.argmax(xax==f2l_temp)
    if xhigh == 0:
        xhigh = np.argmax(xax==f2r_temp)
    
    def complex_fit(x, S0, R, omega, phi, L):
        eta = 1-L
        # sigma = R/((2*np.log(2))**0.5)
        S = S0/(np.pi*R)*(1+((np.pi*np.log(2))**0.5-1)*eta)
        
        A = (R/((R**2+2*(x-omega)**2)))
        D = (x-omega)/((R**2+2*(x-omega)**2))

        gauss = ((4*np.log(2)/np.pi)**0.5)/R*np.exp((-(x-omega)**2*4*np.log(2))/(R**2))
        dysonian = S0/np.pi * (A*np.cos(phi*np.pi/180) - 2*D*np.sin(phi*np.pi/180) + (A*np.sin(phi*np.pi/180) + D*np.cos(phi*np.pi/180)))

        result = (1-L)*gauss + L*dysonian
        return result

    real_data = real_spec/(max(real_spec)-min(real_spec))
    imag_data = 1j * com_spec
    imag_data = imag_data/(max(imag_data)-min(imag_data))

    if component == "real":
        y = real_data
    elif component == "imaginary":
        y = imag_data
    elif component == "complex":
        y = (real_data + imag_data)
    else:
        y = (real_data + imag_data)

    x_data = xax[xhigh:xlow]
    y=y[xhigh:xlow]
    y = y / (max(y)-min(y))

    y_mc = np.sqrt(real_spec**2 + com_spec**2)
    y_mc /= max(y_mc)
    y_mc = y_mc[xhigh:xlow]

    pl = find_peaks(y_mc, prominence=prominence)
    pl = pl[0][0]
    omega = float(x_data[pl])

    #############################
    ## Combined Model Fitting ###
    #############################
    fmodel1 = Model(complex_fit)

    phi_init = -50.

    # Building model parameters
    params = Parameters()
    params.add("S0", min = 1e-9, max = 1.25, value = S0_init, vary=True)
    params.add("R", min = 1e-9, max = 100, value = R_init)
    params.add("omega", value = omega, vary=False)
    params.add("phi", min=-360.0, max=360.0, value=phi_init, vary=True)
    params.add("L", min = 0, max = 1, value = 1, vary=gaussian_component)

    # Run Model
    Comp_Fit = fmodel1.fit(y,params,x=x_data, weights=1/np.sqrt((x_data-omega)**2), method='leastsq', nan_policy='omit', max_nfev=10000)
    y_interp = Comp_Fit.model.func(x_data, **Comp_Fit.best_values)

    model_fits = Comp_Fit.fit_report()

    S0 = Comp_Fit.best_values["S0"]
    R = Comp_Fit.best_values["R"]
    omega = Comp_Fit.best_values["omega"]
    phi = Comp_Fit.best_values["phi"]
    L = Comp_Fit.best_values["L"]
    S = S0/(np.pi*R)*(1+((np.pi*np.log(2))**0.5-1)*(1-L))

    summary = Comp_Fit.summary()
    results = Comp_Fit.params

    dely = Comp_Fit.eval_uncertainty(sigma=3)

    err = results['phi'].stderr

    fig,ax = plt.subplots(1)

    ax.plot(x_data, y, 'ko', label='Complex Data',zorder=1)
    ax.plot(x_data, y_interp, 'r:', linewidth=3, label='Complex Fit',zorder=10)

    plt.fill_between(x_data, Comp_Fit.best_fit-dely, Comp_Fit.best_fit+dely, color="#ABABAB", alpha=0.4, zorder=2, label=r"3-$\sigma$ uncertainty")
    plt.xlabel('$\mathregular{\delta}$('+exp_params['NUC']+") / ppm")
    ax.set_ylabel('Intensity')
    ax.set_xlim(f2p,f1p)
    ax.invert_xaxis
    ax.legend()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    plt.show()

    print("Complex Fit:\n Phi:",phi, "+/–",err, "S:", S, "Omega:",omega, "R:", R, "L:", L)
    fit_results = {"Phi": phi, "Phi Error": err, "S": S, "Omega": omega, "R": R, "Lorentzian Fraction": L}
    return fit_results
