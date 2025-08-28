# SURFACE WAVES ANALYSIS
# A. Carrera - Università degli Studi di Padova

import numpy as np
import numpy as np
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as colors
from tqdm import tqdm


def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_idx = np.where(np.diff(np.signbit(trace)))[0]
    x1 = tt[zc_idx]
    x2 = tt[zc_idx + 1]
    y1 = trace[zc_idx]
    y2 = trace[zc_idx + 1]
    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_idx + 1)
    trace_split = np.split(trace, zc_idx + 1)
    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi


def wiggle_input_check(data, tt, xx, sf, verbose):
    ''' Helper function for wiggle() and traces() to check input

    '''

    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")

    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        if verbose:
            print("tt is automatically generated.")
            print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        if verbose:
            print("xx is automatically generated.")
            print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt must have same as data's rows")
        if verbose:
            print(xx)

    # Input check for streth factor (sf)
    if not isinstance(sf, (int, float)):
        raise TypeError("Strech factor(sf) must be a number")

    # Compute trace horizontal spacing
    ts = np.min(np.diff(xx))

    # Rescale data by trace_spacing and strech_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * ts * sf

    return data, tt, xx, ts


def wiggle(data, tt=None, xx=None, color='k', sf=0.15, verbose=False):
    '''Wiggle plot of a sesimic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran to optimal performance.

    The following color abbreviations are supported:

    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========


    '''

    # Input check
    data, tt, xx, ts = wiggle_input_check(data, tt, xx, sf, verbose)

    # Plot data using matplotlib.pyplot
    Ntr = data.shape[1]

    ax = plt.gca()
    for ntr in range(Ntr):
        trace = data[:, ntr]
        offset = xx[ntr]

        if verbose:
            print(offset)

        trace_zi, tt_zi = insert_zeros(trace, tt)
        ax.fill_betweenx(tt_zi, offset, trace_zi + offset,
                         where=trace_zi >= 0,
                         facecolor=color)
        ax.plot(trace_zi + offset, tt_zi, color)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()


if __name__ == '__main__':
    data = np.random.randn(1000, 100)
    wiggle(data)
    plt.show()

# Normalization function
def normit(tr):
    normalization = np.amax(np.abs(tr), axis=0)
    m, n = tr.shape
    newtr = np.zeros(tr.shape)
    for i in range(n):
        if normalization[i] != 0:
            newtr[:, i] = tr[:, i] / normalization[i]
    return newtr


def cumul_spectra(data, SR, ax=None, xlim=(-10, 300), xlabel="Frequency (Hz)", ylabel="Cumulative power spectra", figsize=(4.5, 3.5)):
    """
    Computes the cumulative power spectrum for a given data array.

    Parameters:
    data (numpy.ndarray): Data array, each column represents a channel
    SR (float): Sampling rate
    xlim (tuple)
    figsize (tuple)

    Returns:
    fig
    mean_power
    median_power
    """
    n = len(data)
    freq = np.fft.fftfreq(n, d=SR)
    positive_freq = freq[freq >= 0]
    potenza_cumulativa = np.zeros(len(positive_freq))

    # Compute the frequency spectrum for each channel
    for i in range(data.shape[1]):
        # Compute the Fourier transform of the i-th channel
        fourier = np.fft.fft(data[:, i])
        
        # Compute the frequency corresponding to each value of the Fourier transform
        frequenze = np.fft.fftfreq(len(data), d=SR)
        
        # Filter negative frequencies
        fourier = fourier[frequenze >= 0]
        frequenze = frequenze[frequenze >= 0]
        
        # Compute the spectral power
        potenza = np.abs(fourier)**2
        
        # Add the spectral power of the i-th channel to the cumulative power
        potenza_cumulativa += potenza

    # Calculate the mean and median of the cumulative power spectrum
    mean_power = np.mean(potenza_cumulativa)
    median_power = np.median(potenza_cumulativa)

    # Plot the cumulative frequency
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(positive_freq, potenza_cumulativa, 'b')
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    TRRabs_norm = (normit(TRRabs.T)).T

    # Add progress bar
    progress_bar = tqdm(total=100, desc='Computing f-k transform', position=0, leave=True)

    # Simulate computation time for demonstration purpose
    for _ in range(100):
        progress_bar.update(1)
        # Simulate computation time
        np.random.random(size=(1000, 1000))

    progress_bar.close()

    return ks, ks_w, frequencies, TRRabs_norm, TRRabs



def plot_fk(TRRabs, ks_w, frequencies, ax=None):
    """
    Plots the f-k spectrum.

    Parameters:
    ks (numpy.ndarray): Wavenumbers.
    frequencies (numpy.ndarray): Frequencies.
    TRRabs (numpy.ndarray): Absolute values of the f-k transform.
    ax (matplotlib.axes.Axes, optional): Axis on which to plot the data. If not provided, a new axis will be created.

    Returns:
    None
    """
    
    color_range = colors.Normalize(vmin=np.min(TRRabs), vmax=np.max(TRRabs))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    im = ax.imshow(TRRabs, extent=[0, max(ks_w), 0, max(frequencies)], aspect='auto', cmap='jet', norm = color_range)
    ax.set_xlabel('k [rad/m]')
    ax.set_ylabel('frequency [Hz]')
    ax.set_title('f-k spectrum')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)

    print('❗❗ Need to fix function cosmetics (axes limits, colorscale) ❗❗')

    plt.show()

    
    

import matplotlib.colors as colors

def plot_and_select_max_amplitudes(TRRabs, ks, frequencies, filename, 
                                   figsize=(12, 8), xlim=None, ylim=None, norm=None):
    """
    Plots the f-k spectrum and allows the user to select maximum amplitudes interactively.

    Parameters:
    TRRabs (numpy.ndarray): Absolute values of the f-k transform.
    ks (numpy.ndarray): Wavenumbers.
    frequencies (numpy.ndarray): Frequencies.
    filename (str): Name of the file for the plot title.
    figsize (tuple): Size of the figure (width, height). Default is (12, 8).
    xlim (tuple): Limits for the x-axis (min, max). Default is None, which uses [0, max(ks)-2.2].
    ylim (tuple): Limits for the y-axis (min, max). Default is None, which uses [0, max(frequencies)].
    norm (colors.Normalize): Normalization for the color scale. Default is None, which uses min and max of TRRabs.

    Returns:
    list: Coordinates of the points selected by the user.
    """
    # Ensure interactive mode is on
    plt.ion()
    
    # Calculate color range normalization based on TRRabs if not provided
    if norm is None:
        norm = colors.Normalize(vmin=np.min(TRRabs), vmax=np.max(TRRabs))
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(TRRabs, extent=[0, max(ks), 0, max(frequencies)], aspect='auto', cmap='jet', norm=norm)
    plt.xlabel('k [rad/m]')
    plt.ylabel('frequency [Hz]')
    plt.title('f-k spectrum: ' + filename)
    plt.colorbar(im, shrink=0.6)
    
    # Set axis limits if provided
    if xlim is None:
        xlim = [0, max(ks) - 2.2]
    plt.xlim(xlim)
    
    if ylim is None:
        ylim = [0, max(frequencies)]
    plt.ylim(ylim)
    
    # Allow user to select points
    print("Click on the max amplitudes and press ENTER when done:")
    c_max = plt.ginput(-1, timeout=0)
    print("Selected points:", c_max)
    
    # Show the plot (for non-interactive environments)
    plt.show(block=True)
    
    # Ensure interactive mode is off
    plt.ioff()
    
    return c_max

