import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from pint import UnitRegistry
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / Path("config.toml")

# Compatibility for TOML loading (Python 3.11+ built-in tomllib, otherwise fallback to toml)
import tomllib

def pi_formatter(val,pos):
    """Formats axis ticks in multiples of pi."""
    val_pi = val / np.pi
    if np.isclose(val_pi, 0):
        return "0"
    elif np.isclose(val_pi, 1):
        return r"$\pi$"
    elif np.isclose(val_pi, -1):
        return r"$-\pi$"
    else:
        # Format cleanly for halves or decimals
        return r"{:.1f}$\pi$".format(val_pi)


def plot_2d_field(data : np.ndarray,
                  x : np.ndarray,
                  y : np.ndarray,
                  xlim : tuple = None,
                  ylim : tuple = None,
                  normalize : bool = True,
                  projection : bool = False,
                  use_pi_ticks : bool = False,
                  use_enl_ticks : bool = False,
                  grid : bool = False,
                  unit_system : str = 'arb. units'
                  ):
    """
    Produces a 2D plot of an array with optional marginal sum-square projections,
    grid lines for every tick, and a slimmed-down top-right colorbar.
    """

    if normalize :
        data = data/np.max(data)
    # 1. Load configuration from TOML if available
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
            
    plot_cfg = config.get("plot_params", {})
    
    # Extract styling parameters with defaults
    fontsize = plot_cfg.get("fontsize", 10)
    cmap = plot_cfg.get("cmap", "seismic")
    figsize = tuple(plot_cfg.get("figsize", (7, 6) if not projection else (8, 8)))
    
    # Grid settings
    if grid :
        grid = plot_cfg.get("grid", True)
    grid_linestyle = plot_cfg.get("grid_linestyle", "--")
    grid_color = plot_cfg.get("grid_color", "gray")
    grid_alpha = plot_cfg.get("grid_alpha", 0.4)
    
    # Update matplotlib rcParams globally for the figure
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': plot_cfg.get("axes_labelsize", fontsize),
        'xtick.labelsize': plot_cfg.get("xtick_labelsize", fontsize),
        'ytick.labelsize': plot_cfg.get("ytick_labelsize", fontsize),
    })

    # 2. Handle Pint units if supplied
    # x_unit_str, y_unit_str = "", ""
    # if isinstance(x, pint.Quantity):
    #     x_unit_str = f" [{x.units:~P}]"
    #     x = x.magnitude
    # if isinstance(y, pint.Quantity):
    #     y_unit_str = f" [{y.units:~P}]"
    #     y = y.magnitude
    match unit_system :
        case '' :
            y_unit_str = ''
            x_unit_str = ''
        case 'arb. units' :
            y_unit_str = ' (arb. units)'
            x_unit_str = ' (arb. units)'
        case _ :
            ureg = UnitRegistry(system = unit_system)
            y_unit_str = " (" + format((1.0* ureg.kg * ureg.m / ureg.s).to_base_units().units,'~P') + ")"
            x_unit_str = " (" + format((1.0 * ureg.m).to_base_units().units,'~P') + ")"

    # 3. Prepare symmetric color scaling
    vmax = np.max(np.abs(data))
    vmin = -vmax

    # Check if x is within [-2pi, 2pi] for pi notation
    # use_pi_ticks = np.all(x >= -2 * np.pi) and np.all(x <= 2 * np.pi)

    # 4. Plotting
    if not projection:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            data, 
            extent=[x[0], x[-1], y[0], y[-1]], 
            origin='lower', 
            aspect='auto', 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax
        )
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(plot_cfg.get("colorbar_label", "Intensity"))
        
        if use_pi_ticks:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
        if use_enl_ticks :
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            
        ax.set_xlabel(f"{plot_cfg.get('xlabel', 'Position')}{x_unit_str}")
        ax.set_ylabel(f"{plot_cfg.get('ylabel', 'Momentum')}{y_unit_str}")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if grid:
            ax.grid(True, which='both', linestyle=grid_linestyle, color=grid_color, alpha=grid_alpha)
            
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 2, 
            width_ratios=[4, 1], 
            height_ratios=[1, 4], 
            wspace=0.08, 
            hspace=0.08
        )
        
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        cbar_ax = fig.add_subplot(gs[0, 1])  # Top-right corner slot
        
        # Main 2D plot
        im = ax_main.imshow(
            data, 
            extent=[x[0], x[-1], y[0], y[-1]], 
            origin='lower', 
            aspect='auto', 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax
        )

        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        
        # Top plot: sum square along axis 0
        sum_sq_x = np.sum(data**2, axis=0)
        if normalize :
            sum_sq_x /= np.max(sum_sq_x)
        ax_top.plot(x, sum_sq_x, color=plot_cfg.get("top_line_color", "black"))
        ax_top.tick_params(labelbottom=False)
        ax_top.fill_between(x,
                            sum_sq_x,
                            color=plot_cfg.get("top_line_color", "black"),
                            alpha=plot_cfg.get("top_fill_alpha", 0.3))
        ax_top.set_ylim((0,np.max(sum_sq_x)*1.1))
        ax_top.set_ylabel(plot_cfg.get("top_ylabel", "Sum Sq."))
        
        # Right plot: sum square along axis 1
        sum_sq_y = np.sum(data**2, axis=1)
        if normalize :
            sum_sq_y /= np.max(sum_sq_y)
        ax_right.plot(sum_sq_y, y, color=plot_cfg.get("right_line_color", "black"))
        ax_right.fill_betweenx(y,
                               sum_sq_y,
                               color=plot_cfg.get("right_line_color", "black"),
                               alpha=plot_cfg.get("right_fill_alpha", 0.3))
        ax_right.set_xlim((0,np.max(sum_sq_y)*1.1))
        ax_right.tick_params(labelleft=False)
        ax_right.set_xlabel(plot_cfg.get("right_xlabel", "Sum Sq."))

        if use_pi_ticks:
            ax_main.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
            ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
            ax_top.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
            ax_top.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
        if use_enl_ticks :
            ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax_right.yaxis.set_major_locator(ticker.MultipleLocator(2))

        ax_main.set_xlabel(f"{plot_cfg.get('xlabel', 'Position')}{x_unit_str}")
        ax_main.set_ylabel(f"{plot_cfg.get('ylabel', 'Momentum')}{y_unit_str}")
        
        # Place colorbar in top-right slot and slim down its width
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label(plot_cfg.get("colorbar_label", "Intensity"), fontsize=fontsize - 1)
        cbar.ax.tick_params(labelsize=fontsize - 2)

        # Shrink and reposition colorbar axes so it isn't overly broad
        pos = cbar_ax.get_position()
        proj_pos = ax_top.get_position()
        new_width = pos.width * 0.15  # Reduce width to ~40% of the grid cell
        cbar_ax.set_position([proj_pos.x0 + proj_pos.width + 0.04, pos.y0, new_width, pos.height]) # - new_width

        # Apply grid to all active subplots if requested
        if grid:
            ax_main.grid(True, which='both', linestyle=grid_linestyle, color=grid_color, alpha=grid_alpha)
            ax_top.grid(True, which='both', linestyle=grid_linestyle, color=grid_color, alpha=grid_alpha)
            ax_right.grid(True, which='both', linestyle=grid_linestyle, color=grid_color, alpha=grid_alpha)

    return fig