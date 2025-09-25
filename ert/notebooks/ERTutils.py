# ELECTRICAL RESISTIVITY TOMOGRAPHY
# A. Carrera - Università degli Studi di Padova

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from pygimli.viewer.mpl import drawStreams, drawSensors
import matplotlib.gridspec as gridspec

def plotDCscheme():
    """
    Disegna schema DC ERT con distribuzione di campo e segnali temporali.
    Parametri predefiniti:
        - 2 cicli completi
        - ON = 4
        - OFF = 4
    """

    # --- setup figure con due colonne ---
    fig = plt.figure(figsize=(8,3))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.], wspace=0.22)

    # --- schema a sinistra con distribuzione di campo ---
    ax0 = plt.subplot(gs[:,0])

    # coordinate elettrodi
    A = np.array([2, 1])   # corrente +
    B = np.array([3, 1])   # corrente -
    M = np.array([6, 1])   # potenziale
    N = np.array([7, 1])   # potenziale
    charges = [(1, A), (-1, B)]

    # griglia sottosuolo (solo sotto y=1)
    nx, nz = 40, 20
    x = np.linspace(0, 10, nx)
    z = np.linspace(-5, 1, nz)
    X, Z = np.meshgrid(x, z)

    # --- campo elettrico ---
    def E_point_charge(q, pos, x, z):
        dx = x - pos[0]
        dz = z - pos[1]
        r3 = (dx**2 + dz**2)**1.5 + 1e-6
        return q * dx / r3, q * dz / r3

    def E_total(x, z, charges):
        Ex, Ez = np.zeros_like(x), np.zeros_like(z)
        for q, pos in charges:
            ex, ez = E_point_charge(q, pos, x, z)
            Ex += ex
            Ez += ez
        return Ex, Ez

    Ex, Ez = E_total(X, Z, charges)

    # disegno superficie
    ax0.plot([0,10],[1,1], 'k', lw=2)
    ax0.streamplot(X, Z, Ex, Ez, color='slategray', density=1.2, linewidth=0.7, arrowsize=1)

    # elettrodi corrente (blu)
    ax0.plot(A[0], A[1], marker='v', markersize=8,
             markerfacecolor='blue', markeredgecolor='k')
    ax0.plot(B[0], B[1], marker='v', markersize=8,
             markerfacecolor='blue', markeredgecolor='k')
    ax0.annotate("Current\ndipole", (2.5,1.5), color="blue", ha="center")

    # elettrodi potenziale (rossi)
    ax0.plot(M[0], M[1], marker='v', markersize=8,
             markerfacecolor='red', markeredgecolor='k')
    ax0.plot(N[0], N[1], marker='v', markersize=8,
             markerfacecolor='red', markeredgecolor='k')
    ax0.annotate("Potential\ndipole", (6.5,1.5), color="tab:red", ha="center")

    ax0.set_xlim(0,10)
    ax0.set_ylim(-7,3)
    ax0.axis("off")

    # --- segnali a destra ---
    # parametri fissi
    n_cycles = 1
    on_duration = 4
    off_duration = 4

    # calcolo sequenza ON/OFF
    block = [0]*off_duration + [1]*on_duration + [0]*off_duration + [-1]*on_duration+ [0]*off_duration
    signal_I = np.tile(block, n_cycles)
    t = np.arange(len(signal_I))

    # potenziale proporzionale
    signal_V = 0.7 * signal_I

    ax1 = plt.subplot(gs[0,1])
    ax1.step(t, signal_I, where="mid", color="blue")
    ax1.set_ylabel("I (mA)", color="blue")
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = plt.subplot(gs[1,1])
    ax2.step(t, signal_V, where="mid", color="tab:red")
    ax2.set_ylabel("u (mV)", color="tab:red")
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    #plt.tight_layout()
    plt.savefig('../figures/DCscheme.png', dpi=150)
    plt.show()
    

def plotABMN(ax, scheme, idx):
    """ Visualize four-point configuration on given axes. """
    def getABMN(scheme, idx):
        """ Get coordinates of four-point cfg with id `idx` from DataContainerERT
        `scheme`."""
        coords = {}
        for elec in "abmn":
            elec_id = int(scheme(elec)[idx])
            elec_pos = scheme.sensorPosition(elec_id)
            coords[elec] = elec_pos.x(), elec_pos.y()
        return coords
    
    coords = getABMN(scheme, idx)
    for elec in coords:
        x, y = coords[elec]
        if elec in "ab":
            color = "green"
        else:
            color = "magenta"
        ax.plot(x, y, marker=".", color=color, ms=10)
        ax.annotate(elec.upper(), xy=(x, y), size=12, ha="center", #fontsize=10, 
                    bbox=dict(boxstyle="round", fc=(0.8, 0.8, 0.8), ec=color), 
                    xytext=(0, 20), textcoords='offset points', 
                    arrowprops=dict(arrowstyle="wedge, tail_width=.5", fc=color, ec=color,
                                    patchA=None, alpha=0.75))
        ax.plot(coords["a"][0])


def show4pointSens(shm, mesh, i=10, idx=None):
    """Show sensitivity for a single four-point measurement."""
    fop = ert.ERTModelling()
    fop.setData(shm)
    mesh.setCellMarkers(pg.Vector(mesh.cellCount(), 1))
    fop.setMesh(mesh)
    model = np.ones(mesh.cellCount())
    fop.createJacobian(model)
    
    sens = fop.jacobian()[i]
    normsens = pg.utils.logDropTol(sens / mesh.cellSizes(), 1e-2)
    normsens /= np.max(normsens)
    
    ax, _ = pg.show(mesh, normsens, cMap="bwr", colorBar=True,
                    label="Sensitivity", nLevs=3, cMin=-1, cMax=1)
    
    plotABMN(ax, shm, i)
    
    # Autoscale axes based on sensors
    xs = [p.x() for p in shm.sensors()]
    ys = [p.y() for p in shm.sensors()]
    ax.set_xlim(min(xs) - .5, max(xs) + .5)
    ax.set_ylim(-5, 1)


def showSensitivity(shm, mesh):
    """Show global sensitivity for all measurements."""
    fop = ert.ERTModelling()
    fop.setData(shm)
    mesh.setCellMarkers(pg.Vector(mesh.cellCount(), 1))
    fop.setMesh(mesh)
    model = np.ones(mesh.cellCount())
    fop.createJacobian(model)

    # Global Sensitivity
    J = fop.jacobian()
    J_np = np.array(J)
    global_sens = np.sum(np.abs(J_np), axis=0)
    normsens = pg.utils.logDropTol(global_sens / mesh.cellSizes(), 1e-2)
    normsens /= np.max(normsens)
    
    ax, _ = pg.show(mesh, normsens, cMap="bwr",
                    label="Global Sensitivity", nLevs=5, cMin=0, cMax=1)
    drawSensors(ax, shm.sensors(), diam=0.2, color='gold', edgecolor='k')
    
    # Autoscale axes based on sensors
    xs = [p.x() for p in shm.sensors()]
    ys = [p.y() for p in shm.sensors()]
    ax.set_xlim(min(xs) - .5, max(xs) + .5)
    ax.set_ylim(-5, max(ys) + 1)


def SurveyStats(df, bins=50):
    """
    Plot an overview of ERT survey data with:
    1. Histogram of apparent resistivity.
    2. Measured voltages vs. measurement index (log y-axis).
    3. Injected currents vs. measurement index (log y-axis).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns:
        - "app" : apparent resistivity (Ω·m)
        - "vp"  : measured voltage (V)
        - "i"   : injected current (A)
    bins : int, optional
        Number of bins for the histogram (default=50).
    """
    # Extract data
    rhoa = df["app"].values
    u = df["vp"].values
    i = df["i"].values
    
    # Take absolute values of voltage if negatives are present
    if np.any(u < 0):
        u = np.abs(u)

    measure_idx = np.arange(len(u))  # index for voltage and current plots

    # Create figure with 3 subplots (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # --- Subplot 1: Histogram of apparent resistivity ---
    axes[0].hist(rhoa, bins=bins, color='skyblue', edgecolor='k')
    axes[0].set_xlabel("Apparent Resistivity (Ω·m)")
    axes[0].set_ylabel("Count")

    # --- Subplot 2: Voltage vs measurement index (log y-axis) ---
    axes[1].scatter(measure_idx, u, color="darkslategray", s=2)
    axes[1].set_xlabel("measure")
    axes[1].set_ylabel("Measured Voltage (mV)")
    axes[1].set_yscale("log")
    axes[1].grid(True, which='both', ls=':', alpha=0.8)

    # --- Subplot 3: Current vs measurement index (log y-axis) ---
    axes[2].scatter(measure_idx, i, color="tab:red", s=2)
    axes[2].set_xlabel("measure")
    axes[2].set_ylabel("Injected Current (mA)")
    axes[2].set_yscale("log")
    axes[2].grid(True, which='both', ls=':', alpha=0.8)

    plt.tight_layout()
    plt.show()



## Synthetic example — fit power-law and linear error laws

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Synthetic data generation ---------------------------------------------
np.random.seed(42)

# true average resistivities (choose log-spaced to cover many decades)
N = 2000
R_avg_true = np.random.choice(np.logspace(-2, 4, 500), size=N)

# define a "true" power-law error model (absolute error)
c_true = 0.01
p_true = 0.15

# generate an error magnitude per pair (with some scatter)
R_err_true = c_true * R_avg_true**p_true
scatter = 0.30  # 20% scatter on the error magnitude
R_err_obs = R_err_true * (1 + scatter * (np.random.randn(N)))

# construct normal and reciprocal measurements so that their absolute difference ~ R_err_obs
# split the difference randomly between the two measurements
half_delta = 0.5 * R_err_obs * (1 + 0.1 * np.random.randn(N))  # small asymmetry
R_normal = R_avg_true + half_delta
R_reciprocal = R_avg_true - half_delta

# compute observed quantities (what you'd get from real data)
R_avg = 0.5 * (R_normal + R_reciprocal)
R_error = np.abs(R_normal - R_reciprocal)

# --- Multibin analysis (20 equal-count bins) --------------------------------
nbins = 20
sort_idx = np.argsort(R_avg)
sorted_R_avg = R_avg[sort_idx]
sorted_R_error = R_error[sort_idx]

# split into equal-count bins
bins = np.array_split(np.arange(N), nbins)

bin_Ravg_mean = np.array([sorted_R_avg[b].mean() for b in bins])
bin_Rerr_mean = np.array([sorted_R_error[b].mean() for b in bins])
bin_count = np.array([len(b) for b in bins])

# --- Fit functions ---------------------------------------------------------
def power_law(x, c, p):
    return c * x**p

def linear_model(x, a, b):
    return a + b * x

# Fit to binned means
popt_pow, pcov_pow = curve_fit(power_law, bin_Ravg_mean, bin_Rerr_mean, p0=[0.002, 0.8])
popt_lin, pcov_lin = curve_fit(linear_model, bin_Ravg_mean, bin_Rerr_mean, p0=[1e-3, 1e-3])

# Compute R^2 for fits on binned points
def r2_score(y, yfit):
    ss_res = np.sum((y - yfit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

bin_fit_pow = power_law(bin_Ravg_mean, *popt_pow)
bin_fit_lin = linear_model(bin_Ravg_mean, *popt_lin)

r2_pow = r2_score(bin_Rerr_mean, bin_fit_pow)
r2_lin = r2_score(bin_Rerr_mean, bin_fit_lin)

# --- Print fit results ----------------------------------------------------
print("Power-law fit: R_err = c * R_avg^p")
print(f"  c = {popt_pow[0]:.6g}, p = {popt_pow[1]:.6g}, R^2 = {r2_pow:.3f}")
print()
print("Linear fit: R_err = a + b * R_avg")
print(f"  a = {popt_lin[0]:.6g}, b = {popt_lin[1]:.6g}, R^2 = {r2_lin:.3f}")

# Also compute implied relative error for a representative range
R_plot = np.logspace(np.log10(sorted_R_avg.min()*0.8), np.log10(sorted_R_avg.max()*1.2), 300)
Rerr_pow_plot = power_law(R_plot, *popt_pow)
Rerr_lin_plot = linear_model(R_plot, *popt_lin)

# --- Plotting --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# Left: absolute error (log-log)
ax[0].scatter(R_avg, R_error, s=8, alpha=0.15)
ax[0].scatter(bin_Ravg_mean, bin_Rerr_mean, s=50, c='C1', edgecolor='k', label='binned means (20 bins)')
ax[0].loglog(R_plot, Rerr_pow_plot, linestyle='--', color='r', linewidth=2, label=f'power-law fit (c={popt_pow[0]:.3g}, p={popt_pow[1]:.3g}, R²={r2_pow:.3f})')
ax[0].loglog(R_plot, Rerr_lin_plot, linestyle=':', linewidth=2, label=f'linear fit (a={popt_lin[0]:.3g}, b={popt_lin[1]:.3g}, R²={r2_lin:.3f})')
ax[0].set_xlabel('R_avg [Ω]')
ax[0].set_ylabel('R_error [Ω]')
ax[0].set_title('Absolute error')
ax[0].legend()
ax[0].grid(True, which='both', ls=':', alpha=0.5)

# Right: relative error (%) (log-log)
rel_pow = (Rerr_pow_plot / R_plot) * 100
rel_lin = (Rerr_lin_plot / R_plot) * 100
ax[1].loglog(R_plot, rel_pow, linestyle='--', color='r', linewidth=2, label='power-law implied relative error')
ax[1].loglog(R_plot, rel_lin, linestyle=':', linewidth=2, label='linear implied relative error')
# plot binned relative errors (points)
ax[1].scatter(bin_Ravg_mean, (bin_Rerr_mean / bin_Ravg_mean) * 100, s=50, c='C1', edgecolor='k', label='binned relative error (%)')
ax[1].set_xlabel('R_avg [Ω]')
ax[1].set_ylabel('Relative error [%]')
ax[1].set_title('Implied relative error')
ax[1].legend()
ax[1].grid(True, which='both', ls=':', alpha=0.5)

plt.tight_layout()
plt.show()



