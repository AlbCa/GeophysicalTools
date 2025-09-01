# ELECTRICAL RESISTIVITY TOMOGRAPHY
# A. Carrera - Università degli Studi di Padova

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert
from pygimli.viewer.mpl import drawStreams, drawSensors


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

