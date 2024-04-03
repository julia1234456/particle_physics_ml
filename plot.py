import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from utils import *


def plot_jets():
    outdir = 'images_out/' 
    assert os.path.isdir(outdir)

    leading_jet_images = np.load(outdir + 'tt_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images = np.asarray(leading_jet_images, dtype=float)

    leading_jet_images0 = np.load(outdir + 'qcd_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images0 = np.asarray(leading_jet_images0, dtype=float)

    std_jet_images0 = list(map(pad_image, leading_jet_images0))
    std_jet_images = list(map(pad_image, leading_jet_images))

    del leading_jet_images0
    del leading_jet_images

    logscale = not True
    logscale = dict(norm=mpl.colors.LogNorm()) if logscale else {}

    for idx in range(5):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        
        for iax, ax in enumerate(axes):
            
            print(std_jet_images0[iax][idx])
            #print(len([std_jet_images0, std_jet_images][iax]))  
            im = ax.imshow([std_jet_images0, std_jet_images][iax][idx], cmap=cmap, **logscale)
            plt.colorbar(im, ax=ax, label = 'Pixel '+ r'$\rho_T$'+ ' (GeV)')
            ax.set_xlabel('Pseudorapidity ' + r'$\eta$')
            ax.set_ylabel('Azimutal angle ' +r'$\phi$' )
            ax.set(title=' '.join((['QCD', 'Top'][iax], 'jet')))

    plt.show()

def plot_jets_normalize():
    outdir = 'images_out/'     
    assert os.path.isdir(outdir)

    leading_jet_images = np.load(outdir + 'tt_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images = np.asarray(leading_jet_images, dtype=float)

    leading_jet_images0 = np.load(outdir + 'qcd_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images0 = np.asarray(leading_jet_images0, dtype=float)

    std_jet_images0 = leading_jet_images0  # list(map(pad_image, leading_jet_images0))
    std_jet_images = leading_jet_images  # list(map(pad_image, leading_jet_images))

    del leading_jet_images0
    del leading_jet_images

    std_jet_images0 = list(map(normalize, std_jet_images0))
    std_jet_images = list(map(normalize, std_jet_images))

    logscale = not True
    logscale = dict(norm=mpl.colors.LogNorm()) if logscale else {}

    for idx in range(3):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        for iax, ax in enumerate(axes):
            im = ax.imshow([std_jet_images0, std_jet_images][iax][idx], cmap=cmap, **logscale)
            plt.colorbar(im, ax=ax, label = 'Pixel '+ r'$\rho_T$'+ ' (GeV)')
            ax.set_xlabel('Pseudorapidity ' + r'$\eta$')
            ax.set_ylabel('Azimutal angle ' +r'$\phi$' )

            ax.set(title=' '.join((['QCD', 'Top'][iax], 'jet')))

    plt.show()

def plot_jets_average():
    outdir = 'images_out/'
    assert os.path.isdir(outdir)

    leading_jet_images = np.load(outdir + 'tt_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images = np.asarray(leading_jet_images, dtype=float)

    leading_jet_images0 = np.load(outdir + 'qcd_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images0 = np.asarray(leading_jet_images0, dtype=float)

    std_jet_images0 = leading_jet_images0  # list(map(pad_image, leading_jet_images0))
    std_jet_images = leading_jet_images  # list(map(pad_image, leading_jet_images))

    del leading_jet_images0
    del leading_jet_images

    std_jet_images0 = list(map(normalize, std_jet_images0))
    std_jet_images = list(map(normalize, std_jet_images))

    fig, axes = plt.subplots(1, 4, figsize=(20, 2.5))
    ims = [0] * len(axes)
    ims[0] = axes[0].imshow(np.average(std_jet_images0, axis=0), cmap=cmap)
    ims[1] = axes[1].imshow(np.average(std_jet_images0, axis=0), norm=mpl.colors.LogNorm(), cmap=cmap)
    ims[2] = axes[2].imshow(np.average(std_jet_images, axis=0), cmap=cmap)
    ims[3] = axes[3].imshow(np.average(std_jet_images, axis=0), norm=mpl.colors.LogNorm(), cmap=cmap)

    for iax, ax in enumerate(axes):
        plt.colorbar(ims[iax], ax=ax)
        ax.set_axis_off()
        ax.set(title='Averaged {} jet image'.format(['QCD', 'Top'][iax // 2]))

    plt.show()


def plot_jets_variance():
    outdir = 'images_out/'
    assert os.path.isdir(outdir)

    leading_jet_images = np.load(outdir + 'tt_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images = np.asarray(leading_jet_images, dtype=float)

    leading_jet_images0 = np.load(outdir + 'qcd_leading_jet.npz', allow_pickle=True)['arr_0']
    leading_jet_images0 = np.asarray(leading_jet_images0, dtype=float)

    std_jet_images0 = leading_jet_images0  # list(map(pad_image, leading_jet_images0))
    std_jet_images = leading_jet_images  # list(map(pad_image, leading_jet_images))

    del leading_jet_images0
    del leading_jet_images

    std_jet_images0 = list(map(normalize, std_jet_images0))
    std_jet_images = list(map(normalize, std_jet_images))

    fig, axes = plt.subplots(1, 4, figsize=(22, 3))
    ims = [0] * len(axes)
    ims[0] = axes[0].imshow(np.var(std_jet_images0, axis=0), cmap=cmap)
    ims[1] = axes[1].imshow(np.var(std_jet_images0, axis=0), norm=mpl.colors.LogNorm(), cmap=cmap)
    ims[2] = axes[2].imshow(np.var(std_jet_images, axis=0), cmap=cmap)
    ims[3] = axes[3].imshow(np.var(std_jet_images, axis=0), norm=mpl.colors.LogNorm(), cmap=cmap)

    for iax, ax in enumerate(axes):
        plt.colorbar(ims[iax], ax=ax)
        ax.set_axis_off()
        ax.set(title='Variance of {} jet image'.format(['QCD', 'Top'][iax // 2]))

    plt.show()