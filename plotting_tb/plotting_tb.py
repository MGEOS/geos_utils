"""
Plotting toolbox.
--------------------------
author: Matthias Gassilloud
date: 03.06.2025
--------------------------
"""



def change_spine(ax, width):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)


def set_size(width=390, fraction=1, ratio='golden'):
    """Set figure dimensions to avoid scaling in LaTeX. Parameters
    ----------
    width: float Document textwidth or columnwidth in pts
    fraction: float, optional Fraction of the width which you wish the figure to occupy
    ratio: str/flaot/int: ratio height/width

    Returns
    -------
    fig_dim: tuple Dimensions of figure in inches
    """

    fig_width_pt = width * fraction  # Width of figure (in pts)
    inches_per_pt = 1 / 72.27  # Convert from pt to inches

    # Golden ratio to set aesthetic figure height https://disq.us/p/2940ij3
    if ratio == 'golden':
        r = (5**.5 - 1) / 2
    elif ratio == 'goldensmall':
        r = (5**.5 - 1) / 3
    elif ratio == 'default':
        r = 6/8  # default ratio
    elif ratio == 'default_smaller':
        r = 5/9  # default smaller ratio
    elif ratio == 'golden_portrait':
        r = 1 + (1 - (5**.5 - 1) / 2)  # default smaller ratio
    elif type(ratio) == float or type(ratio) == int:
        r = ratio

    fig_width_in = fig_width_pt * inches_per_pt  # Figure width in inches
    fig_height_in = fig_width_in * r  # Figure height in inches
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim