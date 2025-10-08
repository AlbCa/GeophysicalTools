# ========================================
# SEISMIC REFRACTION ANALYSIS UTILITIES
# A. Carrera - University of Padova
# ========================================

import os
import numpy as np
import pandas as pd
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
from scipy import signal


datadir = "../data/"
outdir = '../data/fb/'
figdir = '../figures/'

directories = [datadir, outdir, figdir]

for d in directories:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f'📁 Directory created: {d}')


# ===============================
# 1. Preprocessing Utilities
# ===============================

def load_stream(filename, datadir):
    """
    Load seismic data (SEG-Y or SEG2) and extract:
        - data (nsamples × ntraces)
        - sampling rate (SR)
        - receiver positions x (if available)

    Parameters
    ----------
    filename : str
        File name of seismic data.
    datadir : str
        Directory containing the file.

    Returns
    -------
    data : np.ndarray
        Array of shape (nsamples, ntraces) with seismic traces.
    SR : float
        Sampling interval [s].
    x : np.ndarray or None
        Receiver positions along the line, if available.
    """
    filepath = os.path.join(datadir, filename)
    st = read(filepath, unpack_trace_headers=True)
    data = np.array([tr.data for tr in st]).T
    SR = st[0].stats.delta
    x = None

    if filename.lower().endswith((".sgy", ".segy")):
        try:
            x = [tr.stats.segy.trace_header.x_coordinate_of_ensemble_position_of_this_trace
                 for tr in st]
            x = np.array(x)
            print("▶ Receiver positions extracted from SEG-Y header.")
        except Exception as e:
            print("⚠ Could not extract X from SEG-Y headers:", e)
    elif filename.lower().endswith(".dat"):
        try:
            keys = list(st[0].stats.seg2.keys())
            if "RECEIVER_LOCATION" in keys:
                x_list = []
                for tr in st:
                    loc_str = tr.stats.seg2["RECEIVER_LOCATION"]
                    try:
                        x_val = float(loc_str.split()[0])
                    except Exception:
                        x_val = 0.0
                    x_list.append(x_val)
                x = np.array(x_list)
                print("▶ Receiver positions extracted from SEG2 header.")
            else:
                print("⚠ RECEIVER_LOCATION not found in SEG2 header. Provide manually.")
        except Exception as e:
            print("⚠ Error reading SEG2 headers:", e)

    return data, SR, x


def insert_zeros(trace, tt=None):
    """
    Insert zero locations in a trace and time vector at zero-crossings using linear interpolation.

    Parameters
    ----------
    trace : np.ndarray
        Single seismic trace (1D array).
    tt : np.ndarray, optional
        Time vector corresponding to trace samples. If None, uses np.arange(len(trace)).

    Returns
    -------
    trace_zi : np.ndarray
        Trace with zeros inserted at zero-crossings.
    tt_zi : np.ndarray
        Corresponding time vector with zero-crossings inserted.
    """
    if tt is None:
        tt = np.arange(len(trace))

    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    for i in range(len(tt_zero)):
        tt_zi = np.hstack((tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack((trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def normit(data):
    """
    Normalize traces for plotting (divide by max absolute amplitude per trace).

    Parameters
    ----------
    data : np.ndarray
        Seismic data (nsamples x ntraces).

    Returns
    -------
    normalized_data : np.ndarray
        Normalized data array.
    """
    return data / np.max(np.abs(data), axis=0)


# ===============================
# 2. Plot Utilities
# ===============================

def wiggle_input_check(data, tt, xx, sf, verbose):
    """
    Helper function to validate inputs for wiggle plotting.

    Returns preprocessed data, time vector, offsets, and trace spacing.
    """
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")
    if len(data.shape) != 2:
        raise ValueError("data must be 2D")

    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose: print("tt automatically generated.")
    else:
        if type(tt).__module__ != np.__name__ or len(tt.shape) != 1 or tt.shape[0] != data.shape[0]:
            raise ValueError("tt must be a 1D array matching data rows.")

    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose: print("xx automatically generated.")
    else:
        if type(xx).__module__ != np.__name__ or len(xx.shape) != 1 or len(xx) != data.shape[1]:
            raise ValueError("xx must be a 1D array matching data columns.")

    if not isinstance(sf, (int, float)):
        raise TypeError("Stretch factor (sf) must be a number.")

    ts = np.min(np.diff(xx))
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, SR=None, ax=None, color='k', sf=0.15, verbose=False):
    """
    Wiggle plot of seismic data with filled positive and negative amplitudes.

    Parameters
    ----------
    data : np.ndarray
        2D array (time x channels)
    tt : np.ndarray, optional
        Time vector [s]. If None, generated automatically.
    xx : np.ndarray, optional
        Receiver positions. If None, equally spaced.
    SR : float
        Sampling interval [s].
    ax : matplotlib.axes.Axes, optional
        Axis object for plotting.
    color : str
        Fill and trace color.
    sf : float
        Stretch factor for horizontal scaling.
    verbose : bool
        If True, prints debug info.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with wiggle plot.
    """
    if ax is None:
        ax = plt.gca()

    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)
    nsamples, ntraces = data.shape
    tt = np.arange(nsamples) * SR  # ensure time axis in seconds

    for ntr in range(ntraces):
        trace = data[:, ntr]
        offset = xx[ntr]
        trace_zi, tt_zi = insert_zeros(trace, tt)

        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0, facecolor=color, alpha=0.9)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi < 0, facecolor=color, alpha=0.7)
        ax.plot(trace_zi + offset, tt_zi, color=color, linewidth=0.8)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Time (s)")
    return ax


import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 3. picking
# ===============================

def pick_fb(data, SR, x, filename, ax=None, figsize=(10, 6), ylim=None):
    """
    Interactive picking of first arrivals (first breaks) on a wiggle plot.

    Parameters
    ----------
    data : np.ndarray
        Seismic data (nsamples x ntraces)
    SR : float
        Sampling interval [s]
    x : np.ndarray
        Receiver positions [m]
    filename : str
        Name of the seismic file (used for plot title)
    ax : matplotlib.axes.Axes, optional
        Axis to plot into
    figsize : tuple, optional
        Figure size
    ylim : tuple, optional
        Limits for the time axis (e.g. (0.6, 0.0))

    Returns
    -------
    picks : list of tuples
        Clicked coordinates [(x1, t1), (x2, t2), ...]
    x_pick : np.ndarray
        Picked receiver positions [m]
    t_pick : np.ndarray
        Picked times [s]
    ax : matplotlib.axes.Axes
        Axis with plotted picks
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Mostra il wiggle plot
    wiggle(normit(data), SR=SR, xx=x, ax=ax)
    ax.set_title(f"First Break Picking: {filename}")
    ax.set_xlabel("Offset [m]")
    ax.set_ylabel("Time [s]")

    # Applica i limiti sull'asse Y se forniti
    if ylim is not None:
        ax.set_ylim(ylim)

    print("👉 Clicca sui primi arrivi (premi ENTER per terminare)")
    plt.ion()
    picks = plt.ginput(-1, timeout=0)
    plt.ioff()
    print(f"✅ {len(picks)} punti selezionati.")

    # Estrai coordinate (offset, tempo)
    x_pick = np.array([p[0] for p in picks])
    t_pick = np.array([p[1] for p in picks])

    return picks, x_pick, t_pick, ax


def extract_fb(x_pick, t_pick, filename):
    """
    Plot and save first-break picks (travel-time curve).

    Parameters
    ----------
    x_pick : np.ndarray
        Receiver offsets [m].
    t_pick : np.ndarray
        Picked first arrival times [s].
    filename : str
        Original file name for naming outputs.
    datadir : str, optional
        Base directory for saving first-break files (default: "../data/fb/").

    Returns
    -------
    df : pd.DataFrame
        DataFrame with offset and time.
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # travel-time plot
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(x_pick, t_pick, color='tab:blue', alpha=0.8, label='First breaks', zorder=3)
    ax.plot(x_pick, t_pick, color='tab:blue', linewidth=1.5)
    ax.set_xlabel('Offset [m]')
    ax.set_ylabel('Time [s]')
    ax.set_title('Travel-time curve: ' + filename)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # File di output
    outfile = os.path.join(outdir, filename[:-4] + '_fb.txt')
    print(f'📂 First-break be saved in: {outfile}')

    # 
    df = pd.DataFrame({'x(m)': x_pick, 't(s)': t_pick})
    df.to_csv(outfile, sep=',', index=False)

    return df, fig, ax
