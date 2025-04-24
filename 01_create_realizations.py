"""
Diogo H F de Souza - Monday April 23 2025
Generating small deviations to n(z).
TODO: Check normalization of n(z). See for e.g. CSL/number_density/photoz_fisher/photoz_fisher.py `nz /= np.trapz(nz, z)`
"""

import numpy as np
import copy

nz_ref = np.loadtxt("./data_realizations/des_y3_source.nz")

lz = nz_ref.shape[0] ## len(nz) = 300
bins = nz_ref.shape[1] - 1 ## doesn't count the redshift column. DES = 4 bins
total_nz = lz * bins

def make_dataset():
    with open("./data_realizations/des_y3_real.dataset", "r") as ref:
        lines = ref.readlines()
        for col in range(bins):
            for row in range(lz):
                for j in ["left", "right"]:
                    with open("./data_realizations/des_y3_real_bin%d_%s_nz%d.dataset" % (col,j,row), "w") as f:
                        for line in lines:
                            if "nz_source_file = des_y3_source.nz" in line:
                                f.write("nz_source_file = des_y3_source_bin%d_%s_nz%d.nz\n" %(col,j,row))
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
                y_left = np.array([np.stack(np.insert(x_left[i], 0, z[i])) for i in range(lz)]) ## add z
                np.savetxt("./data_realizations/des_y3_source_bin%d_left_nz%d.nz"  % (col,row), y_left) ## saves

        for col in range(bins):
            for row in range(lz):
                x_right = copy.copy(nzs)
                x_right[row][col] = nzs[row][col] + step
                y_right = np.array([np.stack(np.insert(x_right[i], 0, z[i])) for i in range(lz)])
                np.savetxt("./data_realizations/des_y3_source_bin%d_right_nz%d.nz" % (col,row), y_right)
        return 0
    except:
        print("something went wrong when creating source nz files")
        return 1    

# make_sources()