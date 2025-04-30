"""
Diogo H F de Souza Wednesday April 23 2025

Modified from EXAMPLE_EVALUATE1.ipynb
"""


#### LIBS
import sys, platform, os
os.environ['OMP_NUM_THREADS'] = '12'
import matplotlib
import math
from matplotlib import pyplot as plt
import numpy as np
import euclidemu2
import scipy
import cosmolike_lsst_y1_interface as ci
from getdist import IniFile
import itertools
import iminuit
import functools

print(sys.version)
print(os.getcwd())

#### IMPORT CAMB
sys.path.insert(0, os.environ['ROOTDIR']+'/external_modules/code/CAMB/build/lib.linux-x86_64-'+os.environ['PYTHON_VERSION'])
import camb
from camb import model
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#### CAMB AND COSMOLIKE SETTINGS
CAMBAccuracyBoost = 1.1
non_linear_emul = 2

IA_model = 1
IA_redshift_evolution = 3

ntheta = 20 
theta_min_arcmin = 2.5 
theta_max_arcmin = 250 

#### FIDUCIAL COSMOLOGY AND DES SYSTEMATICS
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
w0pwa = -0.9
w = -0.9
mnu = 0.06
DES_DZ_S1 = 0.0414632
DES_DZ_S2 = 0.00147332
DES_DZ_S3 = 0.0237035
DES_DZ_S4 = -0.0773436
DES_M1 = 0.0191832
DES_M2 = -0.0431752
DES_M3 = -0.034961
DES_M4 = -0.0158096
DES_A1_1 = 0.606102
DES_A1_2 = -1.51541
DES_A2_1 = -1.7938938475734911 
DES_A2_2 = -1.5448080290038528
DES_BTA_1 = 0.8154011496506723

#### GET LINEAR/NL POWER SPECTRA AND COMOVING DISTANCE WITH CAMB
def get_camb_cosmology(omegam = omegam, 
                       omegab = omegab, 
                       H0 = H0, 
                       ns = ns, 
                       As_1e9 = As_1e9, 
                       w = w, 
                       w0pwa = w0pwa, 
                       AccuracyBoost = 1.0, 
                       kmax = 10, 
                       k_per_logint = 20, 
                       CAMBAccuracyBoost=1.1):

    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    CAMBAccuracyBoost = CAMBAccuracyBoost*AccuracyBoost
    kmax = max(kmax/2.0, kmax*(1.0 + 3*(AccuracyBoost-1)))
    k_per_logint = max(k_per_logint/2.0, int(k_per_logint) + int(3*(AccuracyBoost-1)))
    extrap_kmax = max(max(2.5e2, 3*kmax), max(2.5e2, 3*kmax) * AccuracyBoost)

    z_interp_1D = np.concatenate( (np.concatenate( (np.linspace(0,2.0,1000),
                                                    np.linspace(2.0,10.1,200)),
                                                    axis=0
                                                 ),
                                   np.linspace(1080,2000,20)),
                                   axis=0)
    
    z_interp_2D = np.concatenate((np.linspace(0, 2.0, 130),np.linspace(2.25, 10, 50)),  axis=0)
    log10k_interp_2D = np.linspace(-4.2, 2.0, 1400)

    pars = camb.set_params(H0=H0, 
                           ombh2=omegabh2(omegab, H0), 
                           omch2=omegach2(omegam, omegab, mnu, H0), 
                           mnu=mnu, 
                           omk=0, 
                           tau=0.06,  
                           As=As(As_1e9), 
                           ns=ns, 
                           halofit_version='takahashi', 
                           lmax=10,
                           AccuracyBoost=CAMBAccuracyBoost,
                           lens_potential_accuracy=1.0,
                           num_massive_neutrinos=1,
                           nnu=3.046,
                           accurate_massive_neutrino_transfers=False,
                           k_per_logint=k_per_logint,
                           kmax = kmax);
    
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');    
    
    pars.NonLinear = model.NonLinear_both
    
    pars.set_matter_power(redshifts = z_interp_2D, kmax = kmax, silent = True);
    results = camb.get_results(pars)
    
    PKL  = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot", nonlinear = False, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    
    PKNL = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot",  nonlinear = True, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    
    lnPL = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    for i in range(len(z_interp_2D)):
        lnPL[i::len(z_interp_2D)] = np.log(PKL.P(z_interp_2D[i], np.power(10.0,log10k_interp_2D)))
    lnPL  += np.log(((H0/100.)**3)) 
    
    lnPNL  = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    if non_linear_emul == 1:
        params = { 'Omm'  : omegam, 
                   'As'   : As(As_1e9), 
                   'Omb'  : omegab,
                   'ns'   : ns, 
                   'h'    : H0/100., 
                   'mnu'  : mnu,  
                   'w'    : w, 
                   'wa'   : wa(w0pwa, w)
                 }
        kbt, bt = euclidemu2.get_boost( params, 
                                        z_interp_2D, 
                                        np.power(10.0, np.linspace( -2.0589, 0.973, len(log10k_interp_2D)))
                                      )
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        
        for i in range(len(z_interp_2D)):    
            lnbt = scipy.interpolate.interp1d(np.log10(kbt), np.log(bt[i]), kind = 'linear', 
                                              fill_value = 'extrapolate', 
                                              assume_sorted = True)(log10k_interp_2D)
            lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0
            lnPNL[i::len(z_interp_2D)]  = lnPL[i::len(z_interp_2D)] + lnbt
    elif non_linear_emul == 2:
        for i in range(len(z_interp_2D)):
            lnPNL[i::len(z_interp_2D)] = np.log(PKNL.P(z_interp_2D[i], np.power(10.0, log10k_interp_2D)))            
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        lnPNL += np.log(((H0/100.)**3))

    G_growth = np.sqrt(PKL.P(z_interp_2D,0.0005)/PKL.P(0,0.0005))
    G_growth = G_growth*(1 + z_interp_2D)
    G_growth = G_growth/G_growth[len(G_growth)-1]
    
    chi = results.comoving_radial_distance(z_interp_1D, tol=1e-4) * (H0/100.)

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi)


#### GET SHEAR POWER SPECTRA
def C_ss_tomo_limber(ell, 
                     omegam = omegam, 
                     omegab = omegab, 
                     H0 = H0, 
                     ns = ns, 
                     As_1e9 = As_1e9, 
                     w = w, 
                     w0pwa = w0pwa,
                     A1  = [DES_A1_1, DES_A1_2, 0, 0], 
                     A2  = [DES_A2_1, DES_A2_2, 0, 0],
                     BTA = [DES_BTA_1, 0, 0, 0],
                     shear_photoz_bias = [DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4],
                     M = [DES_M1, DES_M2, DES_M3, DES_M4],
                     baryon_sims = None,
                     AccuracyBoost = 1.0, 
                     kmax = 10, 
                     k_per_logint = 20, 
                     CAMBAccuracyBoost=1.1,
                     CLAccuracyBoost = 1.0, 
                     CLIntegrationAccuracy = 1):

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi) = get_camb_cosmology(omegam=omegam, 
        omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
        k_per_logint=k_per_logint, CAMBAccuracyBoost=CAMBAccuracyBoost)

    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLSamplingBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(0, CLIntegrationAccuracy + 3*(AccuracyBoost-1.0))
    ci.init_accuracy_boost(1.0, CLSamplingBoost, int(CLIntegrationAccuracy))

    ci.set_cosmology(omegam = omegam, 
                     H0 = H0, 
                     log10k_2D = log10k_interp_2D, 
                     z_2D = z_interp_2D, 
                     lnP_linear = lnPL,
                     lnP_nonlinear = lnPNL,
                     G = G_growth,
                     z_1D = z_interp_1D,
                     chi = chi)
    ci.set_nuisance_shear_calib(M = M)
    ci.set_nuisance_shear_photoz(bias = shear_photoz_bias)
    ci.set_nuisance_ia(A1 = A1, A2 = A2, B_TA = BTA)

    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        ci.init_baryons_contamination(sim = baryon_sims)
        
    return ci.C_ss_tomo_limber(l = ell)


#### GET XI TWO POINT FUNCTION
def xi(ntheta = ntheta, 
       theta_min_arcmin = theta_min_arcmin, 
       theta_max_arcmin = theta_max_arcmin, 
       omegam = omegam, 
       omegab = omegab, 
       H0 = H0, 
       ns = ns, 
       As_1e9 = As_1e9, 
       w = w, 
       w0pwa = w0pwa,
       A1  = [DES_A1_1, DES_A1_2, 0.0, 0.0], 
       A2  = [DES_A2_1, DES_A2_2, 0.0, 0.0],
       BTA = [DES_BTA_1, 0, 0, 0.0, 0.0],     
       shear_photoz_bias = [DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4],
       M = [DES_M1, DES_M2, DES_M3, DES_M4],
       baryon_sims = None,
       AccuracyBoost = 1.0, 
       kmax = 10, 
       k_per_logint = 20, 
       CAMBAccuracyBoost=1.1,
       CLAccuracyBoost = 1.0, 
       CLIntegrationAccuracy = 1):

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi) = get_camb_cosmology(omegam=omegam, 
        omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
        k_per_logint=k_per_logint, CAMBAccuracyBoost=CAMBAccuracyBoost)

    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLSamplingBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(0, CLIntegrationAccuracy + 5*(AccuracyBoost-1.0))
    ci.init_accuracy_boost(1.0, CLAccuracyBoost, int(CLIntegrationAccuracy))
    
    ci.init_binning(int(ntheta), theta_min_arcmin, theta_max_arcmin)
    
    ci.set_cosmology(omegam = omegam, 
                     H0 = H0, 
                     log10k_2D = log10k_interp_2D, 
                     z_2D = z_interp_2D, 
                     lnP_linear = lnPL,
                     lnP_nonlinear = lnPNL,
                     G = G_growth,
                     z_1D = z_interp_1D,
                     chi = chi)
    ci.set_nuisance_shear_calib(M = M)
    ci.set_nuisance_shear_photoz(bias = shear_photoz_bias)
    ci.set_nuisance_ia(A1 = A1, A2 = A2, B_TA = BTA)

    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        ci.init_baryons_contamination(sim = baryon_sims)
        
    (xip, xim) = ci.xi_pm_tomo()    
    return (ci.get_binning_real_space(), xip, xim)



#### PLOT XI
def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None, marker = None, 
                linestyle = None, linewidth = None, ylim = [0.88,1.12], 
                cmap = 'gist_rainbow', legend = None, legendloc = (0.6,0.78), yaxislabelsize = 16, 
                yaxisticklabelsize = 10, xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]], 
                bintextsize = 15, figsize = (12, 12), show = 1, thetashow=[3,250]):
    
    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    if xi_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05}
        )
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0}
        )    

    cm = plt.get_cmap(cmap)

    if not (param is None):
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.3
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), np.max(ylim[1]*np.array(ximax))])
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')
                
                if i == 0:
                    if xi_ref is None:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if xi_ref is None:
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                print(i,j)
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                                # if i==0 and j==0:
                                #     axes[0,0].plot(theta, theta*xip[:,0,0]*10**4, color='k',ls='--', 
                                #                 linewidth=next(linewidthcycler))
                                # if i==0 and j==1:
                                #     axes[1,0].plot(theta, theta*xip[:,1,0]*10**4, color='k',ls=':', 
                                #                 linewidth=next(linewidthcycler))
                                # if i==1 and j==2:
                                #     axes[2,1].plot(theta, theta*xip[:,2,1]*10**4, color='k',ls='-.', 
                                #                 linewidth=next(linewidthcycler))

                            else:
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=3)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, color=cm(x/len(xi)), 
                                    linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=3)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    for x, (theta, xip, xim) in enumerate(xi):
                        if theta != theta_ref:
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), markerfacecolor='None',
                                               marker=next(markercycler),  markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)    
    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        fig.legend(legend, 
                   loc=legendloc,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    if not (show is None):
        fig.show()
    else:
        return (fig, axes)


#### INITIALIZE COSMOLIKE
def init_cosmolike(path = "../../external_modules/data/cocoa_des_y3_nzpca",data_file = "des_y3_real.dataset"):
    ini = IniFile(os.path.normpath(os.path.join(path, data_file)))

    lens_file = ini.relativeFileName('nz_lens_file')
    source_file = ini.relativeFileName('nz_source_file')
    lens_ntomo = ini.int("lens_ntomo")
    source_ntomo = ini.int("source_ntomo")

    ci.initial_setup()

    ci.init_accuracy_boost(1.0, 1.0, int(1))

    ci.init_cosmo_runmode(is_linear = False)

    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=lens_file,
        lens_ntomo=int(lens_ntomo), 
        source_multihisto_file=source_file,
        source_ntomo=int(source_ntomo))

    ci.init_IA( ia_model = int(IA_model), 
                ia_redshift_evolution = int(IA_redshift_evolution))

################################################################
################################################################
############# FISHER PIPELINE PCA SPECIFIC PROJECT #############
################################################################
################################################################

def dataset_realization(path = None, data_file = None):
    with open("%s/%s" %(path,data_file), "r") as f_ref :
        lines = f_ref.readlines()
        with open("%s/%s_realization" %(path,data_file), "w") as f_rel :
            for line in lines:
                if "nz_source_file" in line:
                    nz_source_name = line.replace(" ", "").replace("nz_source_file=","").replace(".nz","").replace("\n","")
                    name_new = "nz_source_file = %s_realization.nz\n" % (nz_source_name)
                    f_rel.write(name_new)
                else:
                    f_rel.write(line)

#     nz_ref = np.loadtxt("%s/%s.nz" % (path,nz_source_name))
#     lz = nz_ref.shape[0] ## len(nz) = 300
#     bins = nz_ref.shape[1] - 1 ## doesn't count the redshift column. DES = 4 bins
#     total_nz = lz * bins
#     z = nz_ref[:,0]
#     nzs = nz_ref[:,1:]
#     steps = [-0.01,0.01]
#     steps_labels = ["left","right"]

#     for step,leri in zip(steps,steps_labels):
#         x = copy.copy(nzs) ## avoid memory issue
#         for col in range(bins):
#             for row in range(lz):
#                 x[row][col] = nzs[row][col] + step ## shift nz at bin=col and z=row
#                 y = np.array([np.stack(np.insert(x[i], 0, z[i])) for i in range(lz)]) ## add z back

#         np.savetxt("./%s/%s_realization.nz"  % (path,nz_source_name), y) ## saves
#         init_cosmolike(path=path, data_file="%s_realization" % (data_file)) ## DHFS - will take data directly from `data` folder, not from `external_modules/data` folder                
#         (theta, xip, xim), xi_new = xi(), []
#         (ntheta, ntomo, ntomo2) = xip.shape
#         for i in range(ntomo):
#             for j in range(ntomo):
#                 if j>=i:
#                     xip_new.append(xip[:,i,j])
#         xip_new=np.array(xip_new)
#         np.savetxt("./projects/cocoa_des_y3_nzpca/xip_%s.txt" % (leri), xip_new)

#     for leri in ["left", "right"]:
#         xip_new = []
#         for tbin in range(4):
#             for nzi in range(300):
#                 data_file = file_eps(tbin=tbin,leri=leri,nzi=nzi)
#                 print(data_file)
#                 init_cosmolike(path=path_eps, data_file=data_file) ## DHFS - will take data directly from `data` folder, not from `external_modules/data` folder                
#                 (theta, xip, xim) = xi()
#                 (ntheta, ntomo, ntomo2) = xip.shape
#                 for i in range(ntomo):
#                     for j in range(ntomo):
#                         if j>=i:
#                             xip_new.append(xip[:,i,j])
#         xip_new=np.array(xip_new)
#         np.savetxt("./projects/cocoa_des_y3_nzpca/xip_%s.txt" % leri, xip_new)
#         print('-----------------------------------')
#         print('------------SAVED XIP_%s-----------' % leri)
#         print('-----------------------------------')

# def compute_derivs():
#     xip_left = np.loadtxt('./projects/cocoa_des_y3_nzpca/xip_left.txt')
#     xip_right = np.loadtxt('./projects/cocoa_des_y3_nzpca/xip_right.txt')

#     (nbins,ntheta) = xip_left.shape ## nbins = number of independend bins
#     deriv = np.zeros((nbins,ntheta))
#     step = 0.01

#     print(nbins,ntheta)
#     for i in range(nbins):
#         print(i)
#         for j in range(ntheta):
#             deriv[i,j] = (xip_right[i,j] - xip_left[i,j]) / (2*step)

#     np.savetxt("./projects/cocoa_des_y3_nzpca/derivs.txt", deriv)