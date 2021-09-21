import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import seaborn as sns
from palettable import cartocolors
from husl import hex_to_husl

colours = cartocolors.qualitative.Safe_10.hex_colors
cpalette = sns.color_palette(colours)
cpalette_light = sns.light_palette(hex_to_husl(colours[1]), input="husl")
colours_rgb = [to_rgb(c) for c in colours]

cmap_qualitative = cartocolors.qualitative.Safe_10.mpl_colormap
cmap_light = sns.light_palette(hex_to_husl(colours[1]), input="husl", as_cmap=True)

pltmarkers = ['o', 'X', 'D', 'P', '^']

fontsizelabels = 26
fontsizeticks = 24

def set_style():
    sns.set_palette(cpalette)
    sns.set_style('whitegrid', {'axes.spines.right': True,
                                'axes.spines.top': True,
                                'axes.edgecolor': 'k',
                                'xtick.color': 'k',
                                'ytick.color': 'k',
                                'grid.color':'0.7',
                                'font.family': 'serif',
                                'font.sans-serif': 'cm',
                                'text.usetex': True})

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