# ========================================
# SURFACE WAVES ANALYSIS UTILITIES
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

from evodcinv import Curve, EarthModel, Layer

datadir = "../data/"
invdir = '../data/inv/'
dcdir = '../data/disp_curves/'
figdir = '../figures/'

directories = [datadir, invdir, dcdir, figdir]

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

def plot_synthmodel(velocity_model, ax=None, length=100.0, nx=200, nz=200, cmap="bone_r"):
    """
    2D synthetic velocity model.
    
    Parameters
    ----------
    velocity_model : np.ndarray
        Columns are [thickness, Vp, Vs, rho] (units: m, m/s, kg/m3).
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure and axis are created.
    length : float
        Lateral extension (in m).
    nx, nz : int
        Number of samples for the meshgrid (dx, dy).
    cmap : str
        Colormap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
        
    x = np.linspace(0, length, nx)
    thicknesses = velocity_model[:, 0]
    vs_vals = velocity_model[:, 2]
    
    # profondità massima (se ultimo layer ha thickness=0 -> halfspace)
    if thicknesses[-1] == 0:
        zmax = np.sum(thicknesses[:-1]) * 3.0
    else:
        zmax = np.sum(thicknesses)
    z = np.linspace(0, zmax, nz)
    
    # griglia
    X, Z = np.meshgrid(x, z)
    vs_grid = np.zeros_like(Z)
    cum_th = np.cumsum(thicknesses)
    
    for k in range(len(thicknesses)):
        top = 0 if k == 0 else cum_th[k-1]
        bottom = cum_th[k] if thicknesses[k] > 0 else zmax
        mask = (Z >= top) & (Z < bottom)
        vs_grid[mask] = vs_vals[k]
    
    # riempi half-space
    vs_grid[Z >= cum_th[-1]] = vs_vals[-1]
    
    # plot
    pcm = ax.pcolormesh(x, z, vs_grid, shading="auto", cmap=cmap)
    ax.invert_yaxis()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Layered model")
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(r'$V_s$ (m/s)')
    
    return fig, ax



def plot_fk(TRRnorm, ks_w, frequencies, ax=None):
    """
    Plot normalized f-k spectrum.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    im = ax.imshow(TRRnorm, extent=[0, max(ks_w), 0, max(frequencies)],
                   aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax.set_xlabel('k [rad/m]')
    ax.set_ylabel('frequency [Hz]')
    ax.set_title('f-k spectrum')
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Amplitude')

    return fig
    
# ===============================
# 3. f-k Analysis
# ===============================

def cumul_spectra(data, SR, ax=None, xlim=(0, 400), xlabel="Frequency (Hz)", ylabel="Amplitude",
                  figsize=(4.5, 3.5), alpha=0.3):
    """
    Computes and plots cumulative power spectrum of data.
    """
    n = len(data)
    freq = np.fft.fftfreq(n, d=SR)
    positive_freq = freq[freq >= 0]
    cum_power = np.zeros(len(positive_freq))

    for i in range(data.shape[1]):
        fourier = np.fft.fft(data[:, i])
        fourier = fourier[freq >= 0]
        cum_power += np.abs(fourier) ** 2

    mean_power = np.mean(cum_power)
    median_power = np.median(cum_power)
    cum_norm = (cum_power - np.min(cum_power)) / (np.max(cum_power) - np.min(cum_power))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(positive_freq, cum_norm, 'b', linewidth=1.2)
    ax.fill_between(positive_freq, 0, cum_norm, color='blue', alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Cumulative Spectral Density")

    return fig, mean_power, median_power


def compute_fk(data, SR, x):
    """
    Computes the f-k transform of the input data

    Parameters:
    data (numpy.ndarray): Input data array
    SR (float): Sampling rate
    x (numpy.ndarray): Array of x-coordinates

    Returns:
    ks (numpy.ndarray): Wavenumbers
    frequencies (numpy.ndarray): Frequencies
    TRRabs_norm (numpy.ndarray): Normalized f-k transform
    TRRabs (numpy.ndarray): Absolute values of the f-k transform
    """
    # Create a Hamming window of size data
    taper = np.hamming(data.shape[1])

    # Compute the wavenumbers and frequencies
    dXmedio = np.mean(np.diff(x))
    NFFT = 2**int(np.ceil(np.log2(data.shape[0])))
    ks = np.linspace(0, 0.5/dXmedio, NFFT)
    ks_w = ks * 2 * np.pi
    frequencies = np.linspace(0, 0.5/SR, NFFT)

    # Apply the taper to the data
    tr2 = data * np.tile(taper, (data.shape[0], 1))

    # Compute the f-k transform
    TRR1 = np.fft.fft2(tr2, (NFFT, NFFT))
    TRR = np.fft.fftshift(TRR1)
    TRRcut = TRR[:NFFT//2+1, NFFT//2:]
    TRRabs = np.abs(TRRcut)
    Theta = np.angle(TRRabs)
    TRRnorm = (TRRabs - np.min(TRRabs)) / (np.max(TRRabs) - np.min(TRRabs))

    # Add progress bar
    progress_bar = tqdm(total=100, desc='Computing f-k transform', position=0, leave=True)

    # Simulate computation time for demonstration purpose
    for _ in range(100):
        progress_bar.update(1)
        # Simulate computation time
        np.random.random(size=(1000, 1000))

    progress_bar.close()

    return ks, ks_w, frequencies, TRRabs, TRRnorm



def pick_amplitudes(TRRnorm, ks, frequencies, filename, ax=None, figsize=(12, 8), ylim=None):
    """
    Plot f-k spectrum and allow interactive picking of maxima.
    Returns resampled wavenumbers, frequencies, and phase velocities.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(TRRnorm, extent=[0, max(ks), 0, max(frequencies)],
                   aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax.set_xlabel('k [rad/m]')
    ax.set_ylabel('frequency [Hz]')
    ax.set_title('f-k spectrum: ' + filename)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Amplitude')
    if ylim is not None:
        ax.set_ylim(ylim)

    print("Click on maxima and press ENTER when done:")
    plt.ion()
    c_max = plt.ginput(-1, timeout=0)
    plt.ioff()
    print("Selected points:", c_max)

    x = [p[0] for p in c_max]
    y = [p[1] for p in c_max]
    n = 50
    ks_resampled = np.linspace(min(x), max(x), n)
    freq_resampled = np.interp(ks_resampled, x, y)
    c = 2 * np.pi * freq_resampled / ks_resampled

    return c_max, ax, ks_resampled, freq_resampled, c


# ===============================
# 4. Dispersion Curve
# ===============================

def extract_dc(freq_sampl, c, filename, datadir="../data/"):
    """
    Plot dispersion curve, save data as text, optionally figure.

    Parameters
    ----------
    freq_sampl : np.ndarray
        Frequencies [Hz].
    c : np.ndarray
        Phase velocities [m/s].
    filename : str
        Original file name for naming outputs.
    datadir : str
        Base directory for 'disp_curves' output.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with frequency and velocity.
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(freq_sampl, c, marker='o', s=20, label='fundamental mode')
    ax.plot(freq_sampl, c, linewidth=2)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('phase velocity [m/s]')
    ax.set_title('dispersion curve: ' + filename)
    ax.legend()
    plt.show()

    outdir = os.path.join(datadir, 'disp_curves', filename[:-4])
    print(f'📁 Dispersion curve will be saved in: {outdir}')

    out = np.column_stack((freq_sampl, c))
    df = pd.DataFrame(out, columns=['freq(Hz)', 'V_ph(m/s)'])
    df.to_csv(os.path.join(outdir, filename[:-4] + '.txt'), sep='\t', index=False)

    return df, fig, ax



def load_dc(path, basename, modes=[0], plot=True):
    """
    Load multiple dispersion curves from txt files (one per mode) and return evodcinv.Curve objects.
    
    Parameters
    ----------
    path : str
        Directory where the files are stored.
    basename : str
        Base filename (without _1, _2 etc.).
    modes : list of int
        Mode numbers to load (0=fundamental, 1=first, ...).
    plot : bool
        If True, plot all loaded dispersion curves.
        
    Returns
    -------
    curves : list of evodcinv.Curve
        List of Curve objects for inversion.
    data_dict : dict
        Dictionary containing frequency and velocity arrays for each mode (for plotting or analysis).
    """
    curves = []
    data_dict = {}
    
    for mode in modes:
        if mode == 0:
            filename = f"{basename}.txt"
            label = "Fundamental"
        else:
            filename = f"{basename}_{mode}.txt"
            label = f"Mode {mode}"
        
        data = np.loadtxt(path + filename, skiprows=1)
        freq, vel = data[:, 0], data[:, 1]
        
        # reorder for evodcinv
        t = 1.0 / freq[::-1]
        v = vel[::-1]
        
        curves.append(Curve(t, v, mode, "rayleigh", "phase"))
        data_dict[mode] = {"frequency": freq, "velocity": vel, "label": label}
    
    if plot:
        fig = plt.figure(figsize=(5,4))
        for mode in modes:
            plt.plot(data_dict[mode]["frequency"], data_dict[mode]["velocity"], '-o',
                     markersize=2, linewidth=1, label=data_dict[mode]["label"])
        plt.title(f"Dispersion curves: {basename}")
        plt.xlabel("Frequency [Hz]") 
        plt.ylabel("Phase velocity [m/s]")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()
    
    return curves, data_dict, t, v


def save_model(res, filename, outdir="../data/inv/"):
    """
    Save the inversion model to a CSV file.

    Parameters
    ----------
    res : evodcinv.EarthModelResult
        Result object from inversion (res.model contains the array to save).
    filename : str
        Base filename for saving.
    outdir : str
        Output directory. A subfolder with 'filename' will be created.
    """
    # Create directory if it does not exist
    save_dir = os.path.join(outdir, filename)
    os.makedirs(save_dir, exist_ok=True)

    # Define header
    header = "d[m], vp[m/s], vs[m/s], rho[g/cm3]"

    # Save model array
    np.savetxt(os.path.join(save_dir, f"{filename}.txt"), res.model,
               header=header, comments='', delimiter=", ", fmt="%.2f")

    print(f'📁 Model saved in: {save_dir}')