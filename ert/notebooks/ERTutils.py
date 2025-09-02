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

