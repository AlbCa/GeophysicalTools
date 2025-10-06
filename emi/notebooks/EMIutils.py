# Electromagnetic Induction Methods (frequency-domain)
# A. Carrera - Università degli Studi di Padova

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import pandas as pd
import sys
from emagpy import Problem

import geopandas as gpd
import contextily as ctx
from pyproj import CRS, Transformer
from matplotlib.ticker import MaxNLocator


def model_output(df):
    """
    Transforms a DataFrame with separate depth and layer columns into a long-form
    DataFrame with columns: x, y, depth, EC.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing 'x', 'y' and pairs of 'depth*' and 'layer*' columns.

    Returns
    -------
    final_df : pandas.DataFrame
        Long-form DataFrame with columns ['x', 'y', 'depth', 'EC'].
    """
    # Extract x and y coordinates
    coordinates = df[['x', 'y']]

    # Initialize empty DataFrame
    final_df = pd.DataFrame()

    # Identify depth and layer columns
    depth_cols = df.filter(like='depth').columns
    layer_cols = df.filter(like='layer').columns

    # Iterate over each depth-layer pair
    for depth_col, layer_col in zip(depth_cols, layer_cols):
        depth_values = df[depth_col]
        layer_values = df[layer_col]

        # Create temporary DataFrame
        temp_df = pd.DataFrame({
            'x': coordinates['x'],
            'y': coordinates['y'],
            'depth': depth_values,
            'EC': layer_values
        })

        # Concatenate
        final_df = pd.concat([final_df, temp_df], ignore_index=True)

    return final_df




def plot_model(df, depths, basemap=True, grid=False,
               utm_crs="EPSG:32632", cmap='turbo', markersize=10,
               axs=None, vmin=None, vmax=None):
    """
    Plot EC (or other scalar field) at specified depths, with optional basemap and grid.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ['x', 'y', 'depth', 'EC'].
    depths : list of float
        Depth values to plot.
    basemap : bool, default=True
        If True, adds a contextily basemap under the scatter plot.
    grid : bool, default=False
        If True, shows a grid on the plots.
    utm_crs : str, default='EPSG:32632'
        CRS of input coordinates (default: UTM zone 32N).
    cmap : str, default='turbo'
        Colormap for EC values.
    markersize : int, default=10
        Size of scatter points.
    axs : matplotlib.axes.Axes or array-like, optional
        Axes object(s) to plot on. If None, new figure and axes are created.
    vmin, vmax : float, optional
        Color scale limits. If None, use automatic scaling from data.
    """
    filtered_df = df[df['depth'].isin(depths)]

    # Convert to GeoDataFrame if basemap is used
    if basemap:
        gdf = gpd.GeoDataFrame(
            filtered_df,
            geometry=gpd.points_from_xy(filtered_df['x'], filtered_df['y']),
            crs=utm_crs
        ).to_crs(epsg=3857)
    else:
        gdf = filtered_df.copy()

    # --- Figure setup ---
    n_depths = len(depths)
    internal_fig = False
    if axs is None:
        fig, axs = plt.subplots(1, n_depths, figsize=(3 * n_depths, 6), sharey=True)
        internal_fig = True
    else:
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = [axs]
        if len(axs) != n_depths:
            raise ValueError(f"Number of provided axes ({len(axs)}) does not match number of depths ({n_depths}).")

    # --- Basemap extent ---
    if basemap:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        dx = (xmax - xmin) * 0.1
        dy = (ymax - ymin) * 0.1
        xlim = (xmin - dx, xmax + dx)
        ylim = (ymin - dy, ymax + dy)
        proj_wm = CRS("EPSG:3857")
        proj_ll = CRS("EPSG:4326")
        project = Transformer.from_crs(proj_wm, proj_ll, always_xy=True).transform

    # --- Plot ---
    for i, depth in enumerate(depths):
        ax = axs[i]
        depth_gdf = gdf[gdf['depth'] == depth]

        scatter_kwargs = dict(cmap=cmap, alpha=0.8, s=markersize)
        if vmin is not None:
            scatter_kwargs['vmin'] = vmin
        if vmax is not None:
            scatter_kwargs['vmax'] = vmax

        if basemap:
            sc = depth_gdf.plot(
                ax=ax, column='EC', legend=False, zorder=2, **scatter_kwargs
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=0)
        else:
            sc = ax.scatter(depth_gdf['x'], depth_gdf['y'], c=depth_gdf['EC'], **scatter_kwargs)

        ax.set_title(f"depth: {depth} m", fontsize=12)
        cbar = plt.colorbar(sc if not basemap else sc.collections[0], ax=ax, shrink=0.3, pad=0.03)
        cbar.set_label(r'$\sigma$ (mS/m)', fontsize=12)

        if basemap:
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            lon, _ = project(xticks, [0]*len(xticks))
            _, lat = project([0]*len(yticks), yticks)

            #ax.set_xticks(xticks)
            #ax.set_yticks(yticks)

            ax.set_xticklabels([f"{val:.5f}°" for val in lon], fontsize=10)
            ax.set_yticklabels([f"{val:.5f}°" for val in lat], fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)
        else:
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)

        if grid:
            ax.grid(True, color='white' if basemap else 'gray', linestyle='--', alpha=0.6)

        if basemap:
            for txt in list(ax.texts):
                if "Esri" in txt.get_text():
                    txt.remove()

    plt.tight_layout()
    if internal_fig:
        plt.show()
