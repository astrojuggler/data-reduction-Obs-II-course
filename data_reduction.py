""" This is a module to use for the data reduction of optical spectroscopy with the designed jupyter notebooks.

The notebooks have been developed for the master course Observational Techniques II at the department of Astronomy, SU.
Teacher: Matthew Hayes
Developer: Mattia Sirressi
Date: November 2020 - March 2021

In this module are included the objects and functions needed to use the notebooks and carry out the steps of data reduciton:
1) bias subtraction, 
2) flatfield normalization, 
3) 1D spectra extraction, 
4) wavelength calibration, 
5) flux calibration, 
6) resampling and coaddition of spectra.

Classes
--------
ImageStats: a class used for saving the image statistics (std, mean, median)

ExtractionWindow: a class used for saving the aperture window to extract a 1D spectrum  
    (xs_min, xs_max, ys, amps, means, stddevs) 

Exceptions
--------
2 errors raised in the funciton get_master_bias if parameters passed are not acceptable


Functions
--------
get_master: computes and returns the master frame combining all the frames of a kind with the method and clip requested.

extract_spectrum: computes spectrum summing pixels along the spatial axis (x) weighted by a gaussian (optimal extraction).

"""

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt  


class ImageStats():
    """A class used for saving the image statistics
    
    Attributes
    -----------
    name: str
        a name describing the statistics of the image intended to be saved
    std: float
        the standard deviation of pixel values of the image
    mean: float
        the mean of pixel values of the image
    median: float
        the median of pixel values of the image

    Methods
    -------
    compute_imstats(data, xmin, xmax, ymin, ymax)
        computes the statistics (std, mean and median) over the specified window of the image 
        
    counts_of_interest(self, data, xmin, xmax, ymin, ymax, threshold)
        counts how many pixels of a frame within a specified window are higher than a given threshold
    """

    def __init__(self, name=None):
        """
        Parameters
        ------------
        name: str, optional
            a name describing the statistics of the image intended to be saved
        """ 
        self.name = name
        
    def compute_imstats(self, data, xmin, xmax, ymin, ymax):
        """A method for computing statistics of a frame within a specified window.
        Defines three new attributes of an object of class image_stats: std, mean and median 

        Parameters
    	-----------
        data: ndarray
            2d array with pixel values of a frame 
    	xmin: float
            lower limit in the x-axis of the desired window for the statistics 
    	xmax: float
            upper limit in the x-axis of the desired window for the statistics 
    	ymin: float
            lower limit in the y-axis of the desired window for the statistics 
    	ymax: float
            upper limit in the y-axis of the desired window for the statistics 
        

        """
        self.std = np.std(data[ymin:ymax, xmin:xmax]) # frame data: y-axis is the first and x-axis is the second  
        self.mean = np.mean(data[ymin:ymax, xmin:xmax])
        self.median = np.mean(data[ymin:ymax, xmin:xmax])

    def counts_of_interest(self, data, xmin, xmax, ymin, ymax, threshold):
        """A method for counting how many pixels of a frame within a specified window are higher than a given threshold.
        Defines a new attributes of an object of class image_stats: px_perc 

        Parameters
    	-----------
        data: ndarray
            2d array with pixel values of a frame 
    	xmin: float
            lower limit in the x-axis of the desired window for the statistics 
    	xmax: float
            upper limit in the x-axis of the desired window for the statistics 
    	ymin: float
            lower limit in the y-axis of the desired window for the statistics 
    	ymax: float
            upper limit in the y-axis of the desired window for the statistics
        threshold: float
            threshold used to compare the pixel values 
            
        Returns
        ----------
        self.mask_high_counts: ndarray
            2d array with boolean pixel value describing a mask of high counts
        

        """
        self.mask_high_counts = np.where(data > threshold, 1, 0)
        px_high_counts = self.mask_high_counts[ymin:ymax, xmin:xmax].sum()
        px_tot = (xmax - xmin) * (ymax - ymin)
        self.px_perc = px_high_counts / px_tot * 100

        return self.mask_high_counts

    
    
class ExtractionWindow():
    """A class used for saving the aperture window to extract the 1D spectrum
    
    Attributes
    -----------
    xs_min: int
        lower limits of the extraction window in the spatial axis (x)
    xs_max: int
        upper limits of the extraction window in the spatial axis (x)
    ys: float
        dispersion coordinates that the other attributes refer to (centres of horizontal stripes)
    amps: ndarray
        amplitudes of the gaussian profiles to weigth the summed pixels along the spatial axis (x) when extracting the spectrum
    means: ndarray
        means of the gaussian profiles to weigth the summed pixels along the spatial axis (x) when extracting the spectrum
    stddevs: ndarray
        stddevs of the gaussian profiles to weigth the summed pixels along the spatial axis (x) when extracting the spectrum
    
    Methods
    -------
    automatic_window(self, data, xlim, ylim)
        automatically detect a window from which to extract the spectrum... idea to be implemented...

    """

    def __init__(self, xs_min, xs_max, ys, amps, means, stddevs):
        """
        Parameters
        ------------
        xs_min: int
            lower limits of the extraction window in the spatial axis (x)
        xs_max: int
            upper limits of the extraction window in the spatial axis (x)
        ys: float
            dispersion coordinates that the other attributes refer to (centres of horizontal stripes)
        amps: ndarray
            amplitudes of the gaussian profiles 
        means: ndarray
            means of the gaussian profiles 
        stddevs: ndarray
            stddevs of the gaussian profiles 
        """ 
        self.xs_min = xs_min
        self.xs_max = xs_max
        self.ys = ys
        self.amps = amps
        self.means = means
        self.stddevs = stddevs


        
# A function to combine the bias frames into the master bias        
def get_master(frames_array, method="mean", n_clip=5):
    """Computes and returns the master frame combining all the frames of a kind with the method and clip requested.

    
    Parameters
    -----------
    frames_array : ndarray
        array of frames of shape (n_frames, ydim, xdim) with their pixel values
    method : string
        specifies the method for combining the frames. Two options available: (1) "mean" or (2) "median"
    n_clip : float
        factor that indicates the threshold above which pixels will be rejected from the combination
        pixels with values distant more than n_clip * std from the mean will be rejected 
        std and mean are computed for each pixel along the array of frames
    
    Returns
    -----------
    master_frame : ndarray
        array of shape (ydim, xdim) representing the master frame, obtained combining all the frames

    Exceptions
    ----------
    Error: if the value of the parameter n_clip is not a positive number
    
    Error: if the string of the parameter method is neither mean nor median    
    """
    
    if n_clip <= 0:
        raise Exception("Error: please choose a positive number for n_clip.") 
    
    std_frame = np.std(frames_array, axis=0)
    mean_frame = np.mean(frames_array, axis=0)
    
    # I don't like this way of clipping very much because it relies on nanmean and nanmedian
    clipped_frames_array = []
    for frame in frames_array:
        clip_frame = np.abs(frame - mean_frame) < n_clip * std_frame
        clipped_frame = np.where(clip_frame, frame, np.nan)
        clipped_frame.reshape((frame.shape))
        clipped_frames_array.append(clipped_frame)
    
    if method == "mean":
        return np.nanmean(clipped_frames_array, axis=0)
    
    elif method == "median":
        return np.nanmedian(clipped_frames_array, axis=0)
    
    else: raise Exception("Error: please specify one of the two following methods for combining the frames: (1) mean or (2) median")



# A function to extract the spectrum from a frame given an aperture window
def extract_spectrum(frame, window):
    """Computes spectrum summing pixels along the spatial axis (x) weighted by a gaussian (optimal extraction).
    
    Parameters
    -----------
    frame : ndarray
        array of shape (ydim, xdim) with their pixel values, representing a spectroscopic frame
    window : class ExtractionWindow
        aperture window to extract the 1D spectrum
    
    Returns
    -----------
    spec_px : ndarray
        array of spectral pixels of the extracted spectrum
    flux : ndarray
        array of the same shape as spec_px with values proportional to the flux
   
    """
    
    ys = window.ys
    
    # Interpolating gaussian properties as a function of dispersion coordinate y with a polynomial of order 3
    polyord = 5
    poly_xs_min = sp.polyfit(ys, window.xs_min, polyord)
    poly_xs_max = sp.polyfit(ys, window.xs_max, polyord)
    poly_amps = sp.polyfit(ys, window.amps, polyord)
    poly_means = sp.polyfit(ys, window.means, polyord)
    poly_stddevs = sp.polyfit(ys, window.stddevs, polyord)
    
    
    # Plotting gaussian properties as a function of dispersion coordinate y and their interpolations
    ymin = int(min(ys))
    ymax = int(max(ys))
    spec_px = np.arange(ymin,ymax)
    
    # making the polynomial arrays callable functions
    f_xs_min = sp.interpolate.interp1d(spec_px, sp.polyval(poly_xs_min, spec_px))
    f_xs_max = sp.interpolate.interp1d(spec_px, sp.polyval(poly_xs_max, spec_px))
    f_amps = sp.interpolate.interp1d(spec_px, sp.polyval(poly_amps, spec_px))
    f_means = sp.interpolate.interp1d(spec_px, sp.polyval(poly_means, spec_px))
    f_stddevs = sp.interpolate.interp1d(spec_px, sp.polyval(poly_stddevs, spec_px))
    
    plt.figure()
    plt.title("Amplitude")
    plt.plot(ys, window.amps, 'o', label="gaussian fit values")
    plt.plot(spec_px, f_amps(spec_px), label="polynomial interpolation")
    plt.xlabel("dispersion coordinate y")
    plt.ylabel("~ flux intensity")
    plt.legend()
    plt.figure()
    plt.title("Central coordinate")
    plt.plot(ys, window.means, 'o', label="gaussian fit values")
    plt.plot(spec_px, f_means(spec_px), label="polynomial interpolation")
    plt.xlabel("dispersion coordinate y")
    plt.ylabel("source position (pix)")
    plt.legend()
    plt.figure()
    plt.title("Standard deviation")
    plt.plot(ys, window.stddevs, 'o', label="gaussian fit values")
    plt.plot(spec_px, f_stddevs(spec_px), label="polynomial interpolation")
    plt.xlabel("dispersion coordinate y")
    plt.ylabel("stdev (pix)")
    plt.legend()
    

    # Extracting the spectrum 
    flux = np.zeros_like(frame[:,0])

    for y in spec_px:
        
        def gauss(x):
            return f_amps(y) * np.exp(-(x-f_means(y))**2/(2*f_stddevs(y))**2)  # guassian at coordinate y
        
        xmin = int(f_xs_min(y))
        xmax = int(f_xs_max(y))
        xrange = np.arange(xmin, xmax)
        weighted_counts = np.array([frame[y,x]*gauss(x) for x in xrange])  # weighted sum at coordinate y
        
        flux[y] = weighted_counts.sum()
        
    # Spectral pixels axis is reversed, so I flip the flux array so that pixel values grow as wavelength grows
    flux_flipped = np.flip(flux[ymin:ymax])    
        
    return spec_px, flux_flipped




