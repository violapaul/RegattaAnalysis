import os

import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt

import numpy.ma as ma
from numba import jit

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr, repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim )
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


data_dir = "/Users/viola/Downloads"

ncdf_ocean = Dataset(os.path.join(data_dir, "ocean_surface.nc"))

nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

# ('u', <class 'netCDF4._netCDF4.Variable'>
#  float64 u(ocean_time, eta_rho, xi_rho)
#      long_name: eastward near-surface velocity
#      units: meter second-1
#      time: ocean_time
#  unlimited dimensions: ocean_time
#  current shape = (37, 1302, 663)
#  filling on, default _FillValue of 9.969209968386869e+36 used),
# ('v', <class 'netCDF4._netCDF4.Variable'>
#  float64 v(ocean_time, eta_rho, xi_rho)
#      long_name: northward near-surface velocity
#      units: meter second-1
#      time: ocean_time
#  unlimited dimensions: ocean_time
#  current shape = (37, 1302, 663)
#  filling on, default _FillValue of 9.969209968386869e+36 used)])


ocean_time = ncdf_ocean.variables['ocean_time'][:]
lon_rho = ncdf_ocean.variables['lon_rho'][:]
lat_rho = ncdf_ocean.variables['lat_rho'][:]

northward = ncdf_ocean.variables['v'][:, :, :]
im = northward[0, :, :]

im[np.isnan(im)] = ma.masked

fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(im)

east, north = G.MAP(lon_rho, lat_rho)
distance_threshold = 30000 # 30 klicks

east_mask = np.logical_and(east < distance_threshold, east > -distance_threshold)
north_mask = np.logical_and(north < distance_threshold, north > -distance_threshold)



def forward_map(east, north, values, threshold, size):
    res = np.zeros((2+size*2, 2+size*2))
    mask = np.ones(res.shape, np.bool)
    forward_map_helper(east, north, values.data, values.mask, res, mask, threshold, size)
    return ma.array(data=res, mask=mask)

@jit(nopython=True)
def forward_map_helper(east, north, values, values_mask, res, mask, threshold, size):
    height, width = east.shape
    for c in range(width):
        for r in range(height):
            if np.abs(east[r, c]) < threshold and np.abs(north[r, c]) < threshold:
                x = int(1 + size + size * east[r, c]/threshold)
                y = int(1 + size + -size * north[r, c]/threshold)
                if not values_mask[r, c]:
                    res[y, x] = values[r, c]
                    mask[y, x] = False

def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    middle = np.int(kernel_size/2)
    for c in range(kernel_size):
        for r in range(kernel_size):
            d = np.square(c - middle) + np.square(r - middle)
            kernel[c, r] = np.exp(-sigma * d/np.square(middle))
    return kernel

def image_interpolate(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, 1.5)
    res = np.zeros(image.shape)
    mask = np.ones(res.shape, np.bool)
    image_interpolate_helper(image.data, image.mask, res, mask, kernel)
    return ma.array(data=res, mask=mask)
                    
@jit(nopython=True)
def image_interpolate_helper(image, mask, res, res_mask, kernel):
    height, width = image.shape
    size = kernel.shape[0]
    delta = np.int(size/2)
    for c in range(delta+1, width-(delta+1)):
        for r in range(delta+1, height-(delta+1)):
            if mask[r, c]:
                s = 0
                w = 0
                for dc in range(-delta, size-delta):
                    for dr in range(-delta, size-delta):
                        if not mask[dr+r, dc+c]:
                            w += kernel[dr, dc]
                            s += image[dr+r, dc+c] * kernel[dr, dc]
                if w > 0:
                    res[r, c] = s/w
                    res_mask[r, c] = False
            else:
                res[r, c] = image[r, c]
                res_mask[r, c] = False

                    
res = forward_map(east, north, im, 40000, 1000)
res.max(), res.min()

fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(res)

rrr = image_interpolate(res, 15)

fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(rrr)



                
fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(im)

im_masked = im.copy()
im_masked.mask = np.logical_and(im.mask, np.logical_not(np.logical_and(east_mask, north_mask)))

fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(im_masked)


fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(np.logical_and(east_mask, north_mask))
