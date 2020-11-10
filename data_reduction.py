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

Exceptions
--------
2 errors raised in the funciton get_master_bias if parameters passed are not acceptable

Functions
--------
get_master_bias: computes and returns the master bias combininf all the bias frames with the method and clip requested.

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
    mean:
        the mean of pixel values of the image
    median:
        the median of pixel values of the image

    Methods
    -------
    compute_imstats(data, xmin, xmax, ymin, ymax)
        Computes the statistics (std, mean and median) over the specified window of the image 
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
        self.std = np.std(data[xmin:xmax, ymin:ymax])
        self.mean = np.mean(data[xmin:xmax, ymin:ymax])
        self.median = np.mean(data[xmin:xmax, ymin:ymax])


# A function to combine the bias frames into the master bias        
def get_master_bias(bias_array, method="mean", n_clip=5):
    """Computes and returns the master bias combininf all the bias frames with the method and clip requested.

    
    Parameters
    -----------
    bias_array : ndarray
        array of bias frames of shape (n_frames, ydim, xdim) with their pixel values
    method: string
        specifies the method for combining the bias frames. Two options available: (1) "mean" or (2) "median"
    n_clip: float
        factor that indicates the threshold above which pixels will be rejected from the combination
        pixels with values distant more than n_clip * std from the mean will be rejected 
        std and mean are computed for each pixel along the array of bias frames
    
    Returns
    -----------
    MasterBias : ndarray
        array of shape (ydim, xdim) representing the master bias frame, obtained combining all the bias frames

    Exceptions
    ----------
    Error: if the value of the parameter n_clip is not a positive number
    
    Error: if the string of the parameter method is neither mean nor median    
    """
    
    if n_clip <= 0:
        raise Exception("Error: please choose a positive number for n_clip.") 
    
    std_frame = np.std(bias_array, axis=0)
    mean_frame = np.mean(bias_array, axis=0)
    
    # I don't like this way of clipping very much because it relies on nanmean and nanmedian
    clipped_bias_array = []
    for bias in bias_array:
        clip_frame = np.abs(bias - mean_frame) < n_clip * std_frame
        clipped_frame = np.where(clip_frame, bias, np.nan)
        clipped_frame.reshape((bias.shape))
        clipped_bias_array.append(clipped_frame)
    
    if method == "mean":
        return np.nanmean(clipped_bias_array, axis=0)
    
    elif method == "median":
        return np.nanmedian(clipped_bias_array, axis=0)
    
    else: raise Exception("Error: please specify one of the two following methods for combining the bias frames: (1) mean or (2) median")








