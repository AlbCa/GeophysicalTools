#!/usr/bin/env python
# coding: utf-8

# ---
# # SurfinWaves - Notebook 1
# This routine reads a real seismic trace and performs the surface waves analysis to extract dispersion curve.
# 
# ##### author: Alberto Carrera

# In[29]:


# import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from scipy import signal
# import functions
import SWAutils as swa

# set directories
datadir = '../data/'
figdir = '../figures/'

#filename = '1006.sg2'
filename = input("Enter the filename {}:")


# In[39]:


# load file
st = read(datadir + filename) # read the SEG-2 file with Obspy
data = np.array([trace.data for trace in st]) # extract the data from all the traces in the file
data = data.T # transpose the data array so that the first axis represents time
print(data.shape)

# flip reverse 'dx' shot
if 'dx' in filename:
    data = np.fliplr(data)
    
#data = data[:, :20]
print(data.shape)
    
# wiggle seismogram
fig = plt.figure(figsize=(7,3.5))
plt.title("shot gather: " +filename[:-4])
swa.wiggle(swa.normit(data))
plt.ylabel('time (trace number)')
plt.xlabel('trace')
#plt.ylim(2500, 0)


# In[40]:


# acquisition scheme
x = np.arange(0,23*3, 3) # receiver # user input !!!!!!!!!!!

# sampling rate (sec)
SR = 0.25e-3
print('The sampling rate is '+str(SR)+' sec')

# view the frequency content of a seismogram (https://community.sw.siemens.com/s/article/what-is-a-power-spectral-density-psd)
fig, ax = plt.subplots(figsize=(4.5,3.5)) 
swa.cumul_spectra(data, SR=SR, ax=ax)
ax.set_title(filename[:-4])
#fig.savefig(figdir + filename[:-4] + "_pwr_spectra.png", bbox_inches='tight', dpi=100, transparent=True)


# In[41]:


# compute f-k and store parameters
ks, ks_w, frequencies, TRRabs_norm, TRRabs = swa.compute_fk(data, SR, x)


# In[42]:


TRRnorm = (TRRabs - np.min(TRRabs)) / (np.max(TRRabs) - np.min(TRRabs))

# Plot
fig, ax = plt.subplots(figsize=(4.5,3.5))
swa.plot_fk(TRRnorm, ks_w, frequencies, ax=ax)
fig.tight_layout()


# In[43]:


fig = plt.figure(figsize=(4.5,3.5))
plt.imshow(TRRnorm, extent=[0, max(ks_w), 0, max(frequencies)], aspect='auto', 
           vmin=0, vmax=1, 
           cmap='jet')
plt.xlabel('k [rad/m]')
plt.ylabel('frequency [Hz]')
plt.title('f-k spectrum: '+filename[:-4])
plt.colorbar(shrink=.5)
plt.ylim([0, 120])
plt.xlim([0, max(ks_w)])
fig.savefig(figdir + filename[:-4] +'.png', bbox_inches='tight',dpi=150)


# In[48]:


import matplotlib
matplotlib.use('TkAgg')

fig = plt.figure(figsize=(12,8))
plt.imshow(TRRnorm, extent=[0, max(ks_w), 0, max(frequencies)], aspect='auto', 
           cmap='jet')
plt.xlabel('k [rad/m]')
plt.ylabel('frequency [Hz]')
plt.title('f-k spectrum')
plt.colorbar()
plt.ylim([0, 120])
plt.xlim([0, max(ks_w)])

# call 'ginput' function to interactive mark on a mpl plot
print("Click on the max amplitudes {}:")
c_max = plt.ginput(-1, timeout=0)
print(c_max)


# In[49]:


# Extract the x and y from the 'c_max' tuple
ks_fund, freq_fund = zip(*c_max) #freq_fund, ks_fund

# resample the picked maxima
x = [p[0] for p in c_max]
y = [p[1] for p in c_max]

n = 50 # samples length
ks = np.linspace(min(x), max(x), n)
ks = np.array(ks) # new ks

freq = np.interp(ks, x, y) # interpolation
freq = np.array(freq) # new freq

# phase velocity
c = 2 * np.pi * freq / ks


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

# First subplot
im1 = ax1.imshow(TRRnorm, extent=[0, max(ks_w), 0, max(frequencies)], aspect='auto',
                 cmap='viridis')
ax1.plot(ks, freq, "r.", label='mode 0', markersize=3)
ax1.set_xlabel('k [rad/m]')
ax1.set_ylabel('frequency [Hz]')
ax1.set_title('f-k spectrum: ' + filename[:-4])
ax1.set_xlim([0, max(ks_w)])
ax1.set_ylim([0, 90])
ax1.legend()
fig.colorbar(im1, ax=ax1, shrink=0.6)

# Second subplot
ax2.scatter(freq, c, marker='.', label='1st mode')
ax2.plot(freq, c, linewidth=1)
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('phase velocity [m/s]')
ax2.set_title('dispersion curve: ' + filename[:-4])
#ax2.ylim([0, 80])
#ax2.set_ylim([100, 260])
ax2.legend()

# Adjust position and size of the second subplot
fig.subplots_adjust(wspace=.01)
pos = ax2.get_position()
pos.x0 = 0.6
pos.x1 = 0.8
pos.y0 = 0.14
pos.y1 = 0.6
ax2.set_position(pos)

plt.show()
fig.savefig(figdir + filename[:-4] +'_1.png', bbox_inches='tight', transparent=True, dpi=150)


# In[51]:


outdir = os.path.join(datadir, 'disp_curves/'+filename[:-4]+'/')
os.makedirs(outdir, exist_ok=True) 
print(f'📁 Dispersion curve will be saved in: {outdir}')

# Save the output file as frequency-velocity
#period = 1/freq
#slowness = 1/c
out = np.column_stack((freq, c))
df = pd.DataFrame(out, columns=['freq(Hz)', 'V_ph(m/s)'])
df.to_csv(outdir+filename[:-4]+'_1.txt', sep='\t', index=False)
df


# In[ ]:




