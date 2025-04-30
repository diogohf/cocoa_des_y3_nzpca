"""
Diogo H F de Souza - Tuesday April 29 2025

A semi apples-to-apples comparison. Using FIDUCIAL_DVv5.0_lingbias27-11-24_Tz_WZ_bqr_0d01_pile3.fits
'semi' because the solely different is derivative method: CosmoSIS (5pt stencil). Here (del xi_p/del n)

Generating small deviations to n(z).
TODO: Check normalization of n(z). See for e.g. CSL/number_density/photoz_fisher/photoz_fisher.py `nz /= np.trapz(nz, z)`
"""

import numpy as np
import copy
from astropy.io import fits

###################################################################
### CONVERT DES DATA FORMAT TO COCOA/COSMOLIKE DATA FORMAT - START
###################################################################

def convert_nz_from_des_to_cocoa_format(dvdes):
    ## Extract source and lens redshift distribution from DES data vector
    try:
        hdul = fits.open(dvdes)
    except FileNotFoundError:    
        print("Error: file %s not found" % (dvdes))
    nz_source_des = np.array(hdul['nz_source'].data)
    nz_lens_des = np.array(hdul['nz_lens'].data)

    ## Map DES nzs to NumPy array
    nz_source_cocoa = np.array([[nz_source_des[j][i] for i in [1,3,4,5,6]] for j in range(1,len(nz_source_des))]) ## Outer loop: starting from 1 to exclude the first redshift where n_source(z)=0 | inner loop: starting from 1 to take the central redshift value.
    nz_lens_cocoa = np.array([[nz_lens_des[j][i] for i in [1,3,4,5,6]] for j in range(1,len(nz_lens_des))]) ## Outer loop: starting from 1 to exclude the first redshift where n_lens(z)=0 | inner loop: starting from 1 to take the central redshift value.

    ## Save w/ CoCoA .nz extension
    np.savetxt('./data_realizations_pile3/des_source_pile3.nz',nz_source_cocoa)
    np.savetxt('./data_realizations_pile3/des_lens_pile3.nz',nz_lens_cocoa)
    return 0        

dvpile3 = './data_realizations_pile3/FIDUCIAL_DVv5.0_lingbias27-11-24_Tz_WZ_bqr_0d01_pile3.fits'
# convert_nz_from_des_to_cocoa_format(dvdes=dvpile3)

###################################################################
### CONVERT DES DATA FORMAT TO COCOA/COSMOLIKE DATA FORMAT - END
###################################################################

nz_ref = np.loadtxt("./data_realizations_pile3/des_source_pile3.nz")

lz = nz_ref.shape[0] ## len(nz) = 299
bins = nz_ref.shape[1] - 1 ## discount the redshift column. DES = 4 bins
total_nz = lz * bins

def make_dataset():
    with open("./data_realizations_pile3/des_real_pile3.dataset", "r") as ref:
        lines = ref.readlines()
        for col in range(bins):
            for row in range(lz):
                for j in ["left", "right"]:
                    with open("./data_realizations_pile3/des_real_bin%d_%s_nz%d_pile3.dataset" % (col,j,row), "w") as f:
                        for line in lines:
                            if "nz_source_file = des_source_pile3.nz" in line:
                                f.write("nz_source_file = des_source_bin%d_%s_nz%d_pile3.nz\n" %(col,j,row))
                            else:
                                f.write(line)

# make_dataset()

z = nz_ref[:,0]
def make_sources():
    nzs = nz_ref[:,1:]
    step = 0.01
    try:
        for col in range(bins):
            for row in range(lz):
                x_left = copy.copy(nzs) ## avoid memory issue
                x_left[row][col] = nzs[row][col] - step ## shift nz at bin=col and z=row
                y_left = np.array([np.stack(np.insert(x_left[i], 0, z[i])) for i in range(lz)]) ## add z back
                np.savetxt("./data_realizations_pile3/des_source_bin%d_left_nz%d_pile3.nz"  % (col,row), y_left) ## saves

        for col in range(bins):
            for row in range(lz):
                x_right = copy.copy(nzs)
                x_right[row][col] = nzs[row][col] + step
                y_right = np.array([np.stack(np.insert(x_right[i], 0, z[i])) for i in range(lz)])
                np.savetxt("./data_realizations_pile3/des_source_bin%d_right_nz%d_pile3.nz" % (col,row), y_right)
        return 0
    except:
        print("something went wrong when creating source nz files")
        return 1    

# make_sources()