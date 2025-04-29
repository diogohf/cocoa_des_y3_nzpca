"""
Diogo H F de Souza - Tuesday April 29 2025

A semi apples-to-apples comparison. Using FIDUCIAL_DVv5.0_lingbias27-11-24_Tz_WZ_bqr_0d01_pile3.fits
'semi' because the solely different is derivative method: CosmoSIS (5pt stencil). Here (del xi_p/del n)

Integrating fisher pipeline with CoCoA
OBS:to run this code:
    cd cocoa_photoz/Cocoa
    conda activate cocoa
    source start_cocoa
then:
    (.local)(cocoa)$ sbatch projects/cocoa_des_y3_nzpca/scripts/run_jacobian_cocoa.sh
"""

#### LIBS
import numpy as np
import matplotlib.ticker as ticker
import fisher_cocoa as fico
import matplotlib.pyplot as plt

path_eps = "/gpfs/projects/MirandaGroup/Diogo/ROMAN-NZ-PROJECT/PZ/cocoa_photoz/Cocoa/projects/cocoa_des_y3_nzpca/data_realizations_pile3" ## `eps` = epsilon on every nz
def file_eps(tbin,leri,nzi):
    """
    `tbin`:: tomographich bin. DES = 0,1,2 or 3
    `nzi` :: n(z[i]) where i = 0,...,299
    `leri`:: left: xi(ni-step), right: xi(ni+step)
    """
    return "des_real_bin%d_%s_nz%d_pile3.dataset" % (tbin,leri,nzi)

def compute_xip():
    for leri in ["left", "right"]:
        xip_new = []
        for tbin in range(4):
            for nzi in range(299):
                data_file = file_eps(tbin=tbin,leri=leri,nzi=nzi)
                print(data_file)
                fico.init_cosmolike(path=path_eps, data_file=data_file) ## DHFS - will take data directly from `data` folder, not from `external_modules/data` folder
                (theta, xip, xim) = fico.xi()
                (ntheta, ntomo, ntomo2) = xip.shape
                for i in range(ntomo):
                    for j in range(ntomo):
                        if j>=i:
                            xip_new.append(xip[:,i,j])
        xip_new=np.array(xip_new)
        np.savetxt("./projects/cocoa_des_y3_nzpca/xip_%s_pile3.txt" % leri, xip_new)
        print('-----------------------------------')
        print('------------SAVED XIP_%s-----------' % leri)
        print('-----------------------------------')

# xip_left = np.loadtxt('./xip_left.txt')
# print(xip_left.shape)

def compute_derivs():
    xip_left = np.loadtxt('./projects/cocoa_des_y3_nzpca/xip_left_pile3.txt')
    xip_right = np.loadtxt('./projects/cocoa_des_y3_nzpca/xip_right_pile3.txt')

    (nbins,ntheta) = xip_left.shape ## nbins = number of independend bins
    deriv = np.zeros((nbins,ntheta))
    step = 0.01 

    print(nbins,ntheta)
    for i in range(nbins):
        print(i)
        for j in range(ntheta):
            deriv[i,j] = (xip_right[i,j] - xip_left[i,j]) / (2*step)

    np.savetxt("./projects/cocoa_des_y3_nzpca/derivs_pile3.txt", deriv)

compute_xip()    ## !! MUST `sbatch` FROM ./cocoa_photoz/Cocoa w/ (.local)(cocoa) ACTIVATED !!
compute_derivs() ## !! MUST `sbatch` FROM ./cocoa_photoz/Cocoa w/ (.local)(cocoa) ACTIVATED !!

def derivs_cocoa_vs_cosmosis():
    derivs_cocoa    = np.loadtxt('./derivs_pile3.txt')
    derivs_cosmosis = np.loadtxt("/gpfs/projects/MirandaGroup/Diogo/ROMAN-NZ-PROJECT/PCA/JACOBIAN/derivatives2.txt")
    derivs_cocoa    = derivs_cocoa.reshape(-1,200)

    fig, ax = plt.subplots(1, 2, figsize=(5, 5),gridspec_kw = {'wspace': 0.1, 'hspace': 0.0})  # 1 row, 3 columns

    fig.gca().annotate("dim = %d x %d" % (derivs_cosmosis.shape[0],derivs_cosmosis.shape[1]), xy=(0.14, 0.353), xycoords='figure fraction',fontsize=10)
    im0 = ax[0].imshow(derivs_cosmosis,cmap="seismic")
    ax[0].set_aspect('auto')
    fig.colorbar(im0,orientation='horizontal',shrink=0.9)
    ax[0].set_title('Jacobian w/ CosmoSIS')

    fig.gca().annotate("dim = %d x %d" % (derivs_cocoa.shape[0],derivs_cocoa.shape[1]), xy=(0.54, 0.353), xycoords='figure fraction',fontsize=10)
    im1 = ax[1].imshow(derivs_cocoa,cmap="seismic")
    ax[1].set_aspect('auto')
    fig.colorbar(im1,orientation='horizontal',shrink=0.9)
    ax[1].yaxis.set_ticks_position('right')
    ax[1].set_title('Jacobian w/ CoCoA')

    plt.savefig("./jacobian_cocoa_pile3.pdf")

# derivs_cocoa_vs_cosmosis()
