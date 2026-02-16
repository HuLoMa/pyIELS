import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import numpy as np

from src.units_contants import v, fs

font = {'family': 'Sans','color':  'black','weight': 'normal','size': 12}

## ============ Plot the Wigner function of a free electron state
## ------ Inputs:
# wigner: Electron Wigner function
# energy_axis: Table containing the energy sampling
# time_axis: Table containing the time sampling
def plot_wigner_electron(wigner, p_axis, z_axis, ene_photon):
    wigner_norm=wigner/np.max(np.max(wigner))
    energy_spectrum=wigner_norm.sum(axis=1)
    energy_spectrum=energy_spectrum/(np.sum(energy_spectrum))                     
    time_structure=wigner_norm.sum(axis=0)

    lamb = (1240/ene_photon)*1e-9
    z_radian=2*np.pi+z_axis*1e-15*(2*np.pi*3e8/lamb)/(v*fs) # Position of the electron in radian

    cmax=np.max(np.max(wigner_norm))
    cmin=-cmax

    sizeoffont=12

    fig=plt.figure()
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[3, 1]) 
    gs.update(wspace=0.05, hspace=0.05)

    # the fisrt subplot
    ax2 = plt.subplot(gs[2])
    im = ax2.imshow(wigner_norm, interpolation='nearest', cmap=cm.seismic,
                   origin='lower', extent=[np.min(z_radian), np.max(z_radian),np.min(p_axis), np.max(p_axis)],
                   vmax=cmax, vmin=cmin,aspect='auto')

    ax2.set_ylim((-15,15))
    ax2.set_xlabel(r'Position $z$ (rad)', fontdict=font,labelpad=5)
    ax2.set_xticks([0,np.pi,2*np.pi,3*np.pi,4*np.pi])
    ax2.set_xticklabels([r'0', r'$\pi$', r'$2\pi$', r'$3\pi$', r' '], fontsize=sizeoffont)
    plt.ylabel(r'Momentum $p_z$', fontdict=font,labelpad=0)
    extraticks2_y=[-10,-5,0,5,10]
    plt.yticks(extraticks2_y)

    #---------------- The right subplot
    # shared axis Y with ax2
    ax3 = plt.subplot(gs[3], sharey = ax2)
    plt.plot(energy_spectrum/np.max(energy_spectrum),p_axis,color='red', alpha=1.00,linewidth=0.3)
    plt.fill_betweenx(p_axis,energy_spectrum/np.max(energy_spectrum), 0, color='red', alpha=.3)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.xlabel(r"$\vert \Psi(p_z) \vert^2 $ (a.u.)", fontdict=font)
    ax3.set_xlim((0,1.1))
    ax3.xaxis.label.set_color('red')

    #---------------- The Top subplot
    # shared axis Y with ax2
    ax0 = plt.subplot(gs[0], sharex = ax2)
    ax0.plot(z_radian,time_structure/np.max(time_structure),color="blue",linewidth=1.5)
    plt.fill_between(z_radian,time_structure/np.max(time_structure), 0, color='blue', alpha=.1)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.ylabel(r"$\vert \Psi(z) \vert^2 $ (a.u.)", fontdict=font,labelpad=20)
    ax0.set_ylim((0,2.0))
    extraticks2_y=[1.]
    plt.yticks(extraticks2_y)
    ax0.yaxis.label.set_color('blue')

    ax1 = plt.subplot(gs[1])
    cbar=fig.colorbar(im, ax=ax1, shrink=1.0,location='left',aspect=10, ticks=[-1, 0, 1])
    # Hide grid lines
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.set_yticklabels(['min', '0', 'max'])  # vertically oriented colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='W (arb. units)', size=12)

    mpl.rc('xtick', labelsize=sizeoffont-1) 
    mpl.rc('ytick', labelsize=sizeoffont-1) 

    plt.tight_layout()
    return gs

## ============ Plot the Wigner function of a photon state
## ------ Inputs:
# wigner: Photon Wigner function
# xvec: Table containing the spatial sampling
# kvec: Table containing the momentum sampling
def plot_wigner_photon(wigner, xvec, kvec):
    K_part=wigner.sum(axis=1)
    X_part=wigner.sum(axis=0)

    xalpha_phasespace=xvec/np.sqrt(2) # Alpha=X+iP/sqrt(2)
    kalpha_phasespace=kvec/np.sqrt(2) # Alpha=X+iP/sqrt(2)

    cmax=np.max(np.max(wigner))
    cmin=-cmax

    sizeoffont=12

    fig=plt.figure(figsize=(5.2 ,5.2))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 2.0], width_ratios=[2.0, 1.0]) 
    gs.update(wspace=0.05, hspace=0.05)

    # the fisrt subplot
    ax2 = plt.subplot(gs[2])
    im = ax2.imshow(wigner, interpolation='nearest', cmap=cm.bwr,
                   origin='lower', extent=[np.min(xalpha_phasespace), np.max(xalpha_phasespace),np.min(kalpha_phasespace), np.max(kalpha_phasespace)],
                   vmax=cmax, vmin=cmin,aspect='auto')
    plt.hlines(0,np.min(xalpha_phasespace),np.max(xalpha_phasespace), colors='k', linestyles='--', lw=0.5)
    plt.vlines(0,np.min(kalpha_phasespace),np.max(kalpha_phasespace), colors='k', linestyles='--', lw=0.5)
    plt.xlabel(r'$\text{Re}(\alpha)$',fontdict=font,labelpad=0)
    plt.ylabel(r'$\text{Im}(\alpha)$',fontdict=font,labelpad=0)

    #---------------- The right subplot
    # shared axis Y with ax2
    ax3 = plt.subplot(gs[3], sharey = ax2)
    plt.plot(K_part/np.max(K_part),kalpha_phasespace,color='red', alpha=1.00,linewidth=1.0)
    plt.fill_betweenx(kalpha_phasespace,K_part/np.max(K_part), 0, color='red', alpha=.3)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.set_xlim((0,1.5))
    extraticks2_x=[]
    plt.xticks(extraticks2_x)
    ax3.xaxis.label.set_color('red')

    #---------------- The Top subplot
    # shared axis Y with ax2
    ax0 = plt.subplot(gs[0], sharex = ax2)
    ax0.plot(xalpha_phasespace,X_part/np.max(X_part),color="blue",linewidth=1.0)
    plt.fill_between(xalpha_phasespace,X_part/np.max(X_part), 0, color='blue', alpha=.1)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylim((0,1.5))
    extraticks2_y=[]
    plt.yticks(extraticks2_y)
    ax0.yaxis.label.set_color('blue')
    plt.yticks(extraticks2_y)

    ax1 = plt.subplot(gs[1])
    cbar=fig.colorbar(im, ax=ax1, shrink=1.0,location='left',aspect=10, ticks=[cmin, 0, cmax])
    # Hide grid lines
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.set_yticklabels(['min', '0', 'max'])  # vertically oriented colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(label='W (arb. units)', fontdict=font)

    mpl.rc('xtick', labelsize=sizeoffont-1) 
    mpl.rc('ytick', labelsize=sizeoffont-1) 

    plt.tight_layout()

    return gs