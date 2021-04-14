import matplotlib.pyplot as plt
import seaborn as sns
from palettable import cartocolors
from husl import hex_to_husl

colours = cartocolors.qualitative.Safe_10.hex_colors
cpalette = sns.color_palette(colours)
cpalette_light = sns.light_palette(hex_to_husl(colours[1]), input="husl")

cmap_qualitative = cartocolors.qualitative.Safe_10.mpl_colormap
cmap_light = sns.light_palette(hex_to_husl(colours[1]), input="husl", as_cmap=True)
cmap_diverging = sns.diverging_palette(h_neg=227, h_pos=4, s=73, l=60, n=12, as_cmap=True)
cmap_diverging.set_bad('white')

pltmarkers = ['o', 'X', 'D', 'P', 'X']

def set_style():
    sns.set_palette(cpalette)
    sns.set_style('darkgrid', {'axes.spines.right': False,
                               'axes.spines.top': False,
                               'axes.facecolor': '.93'})

    plt.rcParams.update({
        'font.family': 'serif',
        'font.sans-serif': 'cm',
        'text.usetex': True,
        'font.size': 14,

        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,

        'savefig.dpi': 75,

        'figure.autolayout': False,
        'figure.figsize': (13, 7),
        'figure.titlesize': 20,

        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'legend.fontsize': 14
    })