#!/usr/bin/env python
# coding: utf-8

# # Automatic Microscopic Image Analysis by Moving Window Local Fourier Transform and Machine Learning
#
# ## Benedykt R. Jany
# ## benedykt.jany[at]uj.edu.pl
#
#
# # Date: 11.2019
#
#
# # Institute of Physics Jagiellonian University in Krakow, Poland
#
#
#
# ## To run this notebook first you have to install HyperSpy https://hyperspy.org/
#


import sys

# if len(sys.argv) == 1:
#     print("missing filename for analysis")
#     print("basic usage: program.py filename auto")
#     print("automatically determine number of components for NMF decomposition")
#     print("\nbasic usage: program.py filename n")
#     print("select n components for NMF decomposition")
#     print("example: program.py filename.tif 7")
#     print("\noptional usage: program.py filename n elementsize xstep ystep rescale")
#     print("set elementsize xstep ystep variables")
#     print("write default for default value")
#     print("rescale - rescale image width to 2048pixels")
#     print("example: program.py filename.tif 7 default default 128")
#     print("only sets ystep=128")
#     print("\ndefault: elementsize = 128 optimal for ~2000x2000 pixels image")
#     print("default: xstep = 64 optimal for ~2000x2000 pixels image")
#     print("default: xstep = 64 optimal for ~2000x2000 pixels image")
#     print("default: do not rescale image width to 2048pixels")
#     sys.exit()

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib

matplotlib.use('Agg')
import hyperspy.api as hs

hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()

# # Load image for analysis

# In[2]:

import imageio  # supported file types https://imageio.readthedocs.io/en/stable/formats.html
import os

# filename = sys.argv[1]

filename = 'data/cubic/defective/images/JEOL BF 50_SIH05 no annotation.tif'
import time

start_time = time.time()

filetype = os.path.splitext(filename)[-1]

if filetype == '.jpg' or filetype == '.png' or filetype == '.bmp':  # color images
    imdata = imageio.imread(filename, as_gray=True)
elif filetype == '.tif' or filetype == '.tiff':  # grayscale images
    imdata = imageio.imread(filename)
else:
    print("not supported file type")
    sys.exit()

im = hs.signals.Signal2D(imdata)

im.metadata.General.title = filename

print("\nAnalyzing", filename)

# In[3]:
# Rescale Image to 2048 width


ImageRescale = False

if len(sys.argv) >= 7 and sys.argv[6] != "default":
    ImageRescale = True

if ImageRescale == True:
    print("rescaling image width to 2048pixels")
    imshape = im.axes_manager.signal_shape
    wscale = imshape[0] / 2048.
    im = im.rebin(scale=[wscale, wscale])
    imdata = im.data

# In[4]:


imdata = im.data  # get image data as NumPy array

# In[5]:


im.plot(scalebar=False, axes_off=True)

# # Moving Window via NumPy as_strided

# ## divide image into parts of elementsize and move it in x and y direction by xstep and ystep

# In[6]:


import numpy as np
from numpy.lib.stride_tricks import as_strided

# In[7]:


elementsize = 128  # 128 default for ~2000x2000 pixels image

if len(sys.argv) >= 4 and sys.argv[3] != "default":
    elementsize = int(sys.argv[3])

# In[8]:


ws = np.arange(elementsize * elementsize).reshape(elementsize,
                                                  elementsize)  # shape of the elements on which you want to perform the operation (e.g. Fourier Transform)

# In[9]:


xstep = 64  # 64 default for ~2000x2000 pixels image
ystep = 64  # 64 default for ~2000x2000 pixels image

if len(sys.argv) >= 5 and sys.argv[4] != "default":
    xstep = int(sys.argv[4])

if len(sys.argv) >= 6 and sys.argv[5] != "default":
    ystep = int(sys.argv[5])

print("Using \telementsize(es)=", elementsize, "\txstep(xs)=", xstep, "\tystep(ys)=", ystep)

# In[10]:


imdataW = as_strided(imdata, shape=(
int((imdata.shape[0] - ws.shape[0] + 1) / xstep), int((imdata.shape[1] - ws.shape[1] + 1) / ystep), ws.shape[0],
ws.shape[1]), strides=(imdata.strides[0] * xstep, imdata.strides[1] * ystep, imdata.strides[0], imdata.strides[1]))

# In[11]:


# imWindow = hs.signals.Signal2D(imdataW)


# In[12]:


# imWindow.plot(cmap='plasma', axes_ticks=False, scalebar=False, axes_off=True) #plot divided image


# # Compute Hanning Window Power Spectrum (FFT) from Local Window Data

# In[13]:


hanningf = np.hanning(elementsize)
hanningWindow2d = np.sqrt(np.outer(hanningf, hanningf))

# In[14]:


imdataWfft = np.fft.fftshift(np.abs(np.fft.fft2(hanningWindow2d * imdataW)) ** 2, axes=(2, 3))

# In[15]:


imdataWfft = imdataWfft + 10000  # adding offset to prevent 0

# In[16]:


imdataWfft = np.log(np.abs(imdataWfft))

# In[17]:


imWindowFFT = hs.signals.Signal2D(imdataWfft)

# In[18]:


# imWindowFFT.plot(cmap='plasma', axes_ticks=False, scalebar=False, axes_off=True) #plot 4D local FFT data


# # Now Machine Learning on Local FFT 4D Data

# ## Perform PCA to determine number of components in the Local FFT data from Scree Plot

# ### Look for inflection point in Scree Plot

# In[19]:


imWindowFFT.decomposition()

# In[20]:


# imWindowFFT.plot_explained_variance_ratio(n=30, xaxis_type='number') #plot PCA Scree Plot


# In[21]:


screedata = imWindowFFT.get_explained_variance_ratio().data

# ## Automatically Analyze PCA Scree Plot

# ### compute gradient on Scree Plot Data

# In[22]:


grad = np.gradient(screedata)

# In[23]:


# gradS = hs.signals.Signal1D(grad)


# In[24]:


# gradS.plot()


# ### find local maxima of gradient

# In[25]:


from scipy.signal import argrelextrema

# In[26]:


gradLocalMaxima0 = argrelextrema(grad, np.greater)
gradLocalMaxima = [x + 1 for x in gradLocalMaxima0[0]]  # add 1 due to the array indexing from 0

# In[45]:


print("Estimated candidates for number of components from PCA Scree Plot\n", gradLocalMaxima)

# In[28]:


NComponents = gradLocalMaxima[0]

if len(sys.argv) >= 3 and sys.argv[2] != "auto":
    NComponents = int(sys.argv[2])

# In[48]:


print("Taking", NComponents, "components for NMF Decomposition")  # estimated number of components (first local maximum)

# In[30]:


# imWindowFFT.plot_explained_variance_ratio(n=30, xaxis_type='number') #plot PCA Scree Plot
imWindowFFT.plot_explained_variance_ratio(n=30, xaxis_type='number', threshold=NComponents,
                                          hline=True)  # plot PCA Scree Plot

# ## Perform Decomposition on the Local FFT data by NMF

# ### you have to provide number of components in output_dimension (e.g. from PCA Scree Plot)

# In[31]:


imWindowFFT.decomposition(algorithm="nmf", output_dimension=NComponents)

# In[32]:


# imWindowFFT.plot_decomposition_results()

# FFT Measurements (optional)#distantace in lengths units = 1/(distance in FFT)*elementsize*scale [unit i.e nm]elementsize*im.axes_manager[0].scale # if properly calibration read from image file
# # Nice color plotting

# ## Loadings

# In[33]:


loadingsS = imWindowFFT.get_decomposition_loadings()

# In[34]:


hs.plot.plot_images(loadingsS, cmap='viridis', scalebar=None, axes_decor='off')

# ## Factors

# In[35]:


factorsS = imWindowFFT.get_decomposition_factors()

# In[36]:


hs.plot.plot_images(factorsS, cmap='plasma', scalebar=None, axes_decor='off')

# # Export Data

# In[37]:

filebase = os.path.splitext(filename)[0]

filebase = filebase + "-es" + str(elementsize) + "-xs" + str(xstep) + "-ys" + str(ystep)

if ImageRescale == True:
    filebase = filebase + "-Rescaled"

if ImageRescale == False:
    filebase = filebase + "-Original"

# In[38]:

print("saving NMF loadings as:\t", filebase + "-Loadings_NMF" + str(NComponents) + ".tif")
loadingsS.save(filebase + "-Loadings_NMF" + str(NComponents) + ".tif",
               overwrite=True)  # tif for ImageJ/FIJI open via BioFormats

# In[39]:

print("saving NMF factors as:\t", filebase + "-Factors_NMF" + str(NComponents) + ".tif")
factorsS.save(filebase + "-Factors_NMF" + str(NComponents) + ".tif",
              overwrite=True)  # tif for ImageJ/FIJI open via BioFormats

# In[40]:


# imWindowFFT.save(filebase+"-Loadings_NMF"+str(NComponents)+"Data")


# # Create PDF Report File

# In[41]:

print("saving report as:\t", filebase + "-NMF" + str(NComponents) + "-Report" + ".pdf")

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

pdf = matplotlib.backends.backend_pdf.PdfPages(filebase + "-NMF" + str(NComponents) + "-Report" + ".pdf")

for fig in range(1, plt.gcf().number + 1):
    pdf.savefig(fig)
pdf.close()

print("Analysis time", "--- %s seconds ---" % (time.time() - start_time))
print("End of Analysis\n")
