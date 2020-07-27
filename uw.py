"""
# UW Live Ocean Data

Found a potentially great resource at UW:

http://faculty.washington.edu/pmacc/LO/LiveOcean.html

> LiveOcean works a lot like the weather forecast models that we all rely on every day. It takes in information about the state of the ocean, atmosphere and rivers on a given day, and then uses the laws of physics (and a large computer) to predict how the state of the ocean in our region will change over the next few days. The things that the model predicts are currents, salinity, temperature, chemical concentrations of nitrate, oxygen, carbon, and biological fields like phytoplankton, zooplankton, and organic particles. It does this in three dimensions, and allowing continuous variation over the full 72 hour forecast it makes every day.

- [How the Model Works](http://faculty.washington.edu/pmacc/LO/how_it_works.html)

> The model framework we use is called the Regional Ocean Modeling System (ROMS) which is a large, flexible computer code that is used by coastal and estuarine oceanographers around the world to simulate different regions. It is known to have excellent numerical properties, such as low numerical diffusion, many choices for turbulence parameterization and open boundary conditions, built-in biogeochemical models, and a large, active user community. We have been using ROMS for realistic simulations in the region for research projects for the past decade and have found it to be a robust tool.

Its based on the [ROMS Finite Element Model](https://www.myroms.org/)

- [Awesome Accessible Description of Tides](http://faculty.washington.edu/pmacc/LO/tides_background.html)

## Data From Parker MacCready

Exchanged emails with Prof. MacCready and he pointed me toward his daily model outputs.

"""

#### Cell #1 Type: module ######################################################

# Basics
import os
import datetime
import itertools as it

import arrow
import time
import numpy as np
import matplotlib.pyplot as plt

import numpy.ma as ma
from numba import jit

import netCDF4

#### Cell #2 Type: module ######################################################

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables
import race_logs                # load data from races
import chart
import utils
import process as p
from utils import DictClass

import metadata

import nbutils
from nbutils import display_markdown, display

#### Cell #3 Type: module ######################################################

G.init_seattle()

#### Cell #4 Type: module ######################################################

# The NetCDF file containing the UW Live Ocean current model can be downloaded as follows.

LIVE_OCEAN_BASE = "https://pm2.blob.core.windows.net"
LIVE_OCEAN_FILE = "ocean_surface.nc"
LIVE_OCEAN_DATA_DIRECTORY = "/Users/viola/BigData"

def live_ocean_path(date_string):
    "Path to the live ocean file for this date."
    adt = arrow.get(date_string)
    result_filename = adt.format("YYYY-MM-DD") + "_ocean_surface.nc"
    return os.path.join(LIVE_OCEAN_DATA_DIRECTORY, result_filename)

def fetch_live_ocean_model(date_string):
    """
    Fetch the UW Live Ocean current model.  If the file has already been downloaded, then
    return the path.  If not, beging an asynchronous download and return False.
    """
    time = utils.time_from_string(date_string)
    path = live_ocean_path(date_string)
    if os.path.exists(path):
        G.logger.info(f"File {path} already downloaded. Skipping.")
        return DictClass(path=path, date=date_string)
    else:
        # Construct the full URL 
        directory = "f" + utils.time_to_string(time, "YYYYMMDD")
        url = f"{LIVE_OCEAN_BASE}/{directory}/{LIVE_OCEAN_FILE}"
        download_path = path + ".incomplete"
        log_file = "/tmp/wget-log"
        command = f"(wget {url} -O {download_path} -c -o {log_file}; mv {download_path} {path})&"
        print(command)
        utils.run_system_command(command)
        display(f"Downloading model to {path}.  Check {log_file}.  Incomplete results here: {download_path}.")
        return False

def read_live_ocean_model(date):
    """
    Read the UW live ocean model for a given date.

    Note this is the date when the model was generated, and includes predicitions for
    subsequent times/dates.  When looking for predictions, it may be necessary to fetch
    the most recent model and then query the future.
    """
    model = fetch_live_ocean_model(date)
    if model is None:
        display("Wait and try again soon!")
        return None
    model.netcdf = netCDF4.Dataset(model.path)

    model.current_n = model.netcdf.variables['v'][:, :, :]
    model.current_e = model.netcdf.variables['u'][:, :, :]

    model.wind_n = model.netcdf.variables['Vwind'][:, :, :]
    model.wind_e = model.netcdf.variables['Uwind'][:, :, :]

    model.ocean_time = model.netcdf.variables['ocean_time'][:]
    model.lon_rho = model.netcdf.variables['lon_rho'][:]
    model.lat_rho = model.netcdf.variables['lat_rho'][:]
    model.east, model.north = G.MAP(model.lon_rho, model.lat_rho)  # Works great with numpy masked arrays.
    return model

def find_times(uw_model, start, finish):
    tstart = utils.time_from_string(start)
    tfinish = utils.time_from_string(finish)
    time_indices = []
    for i, tstamp in enumerate(uw_model.ocean_time):
        t = utils.time_from_timestamp(tstamp)
        if utils.time_after(tstart, t) and utils.time_after(t, tfinish):
            time_indices.append(i)
    if len(time_indices) == 0:
        return time_indices
    if time_indices[0] > 0:
        time_indices = [time_indices[0]-1] + time_indices
    if time_indices[-1] < len(uw_model.ocean_time)-1:
        time_indices = time_indices + [time_indices[-1]+1]
    return time_indices, [utils.time_from_timestamp(uw_model.ocean_time[t]) for t in time_indices]


def region_from_marks(marks, lat_border=0.2, lon_border=0.3):
    pos = [G.STYC_RACE_MARKS[m] for m in marks]
    lats = [p.lat for p in pos]
    lons = [p.lon for p in pos]    
    lat_max, lat_min = chart.max_min_with_border(np.array(lats), lat_border)
    lon_max, lon_min = chart.max_min_with_border(np.array(lons), lon_border)
    return DictClass(lat_max=lat_max, lat_min=lat_min,
                     lon_max=lon_max, lon_min=lon_min)

def lor(a, *b):
    "Numpy logical-or of all arguments." 
    res = a
    for e in b:
        res = np.logical_or(res, e)
    return res

def land(a, *b):
    "Numpy logical-or of all arguments." 
    res = a
    for e in b:
        res = np.logical_and(res, e)
    return res

def display_currents(uw_model, region, time_index):
    """
    Draw a chart overlayed with the UW live ocean current predictions.
    """
    ch = chart.create_chart(region)
    ch.fig = plt.figure(figsize=(8, 10))
    ch.ax = ch.fig.add_subplot(111)   
    ch = chart.draw_chart(ch, ch.ax)
    return draw_current(ch, uw_model, time_index)


def draw_current(ch, uw_model, time_index):

    # Current is in meters/sec. And we typically think in knots.  1 m/s is 2 kts.  If you
    # scale by 1000 then a 1kt current is 500m.

    scale = 1000
    u = scale * uw_model.current_e[time_index, :, :]
    v = scale * uw_model.current_n[time_index, :, :]

    dt = utils.time_from_timestamp(uw_model.ocean_time[time_index])
    ch.datetime = dt

    # Note, for masked arrays mask is TRUE for undefined.
    lat_mask = lor(uw_model.lat_rho > ch.lat_max, uw_model.lat_rho < ch.lat_min)
    lon_mask = lor(uw_model.lon_rho > ch.lon_max, uw_model.lon_rho < ch.lon_min)
    ll_mask = lor(lat_mask, lon_mask)

    mask = np.logical_not(lor(ll_mask, u.mask, v.mask))

    one_knot = (1/G.MS_2_KNOTS) * scale/np.sqrt(2)
    ch.ax.arrow(2700, 737, one_knot, one_knot, head_width=100, length_includes_head=True, color='red')

    ch.ax.quiver(uw_model.east[mask], uw_model.north[mask], u[mask], v[mask], 
                 angles='xy', scale_units='xy', scale=1, color='blue')
    
    ch.ax.set_title(uw_model.date + " : " + utils.time_to_string(dt))
    return ch

def plot_marks(ch, marks):
    "Plot the STYC marks identified by name."
    pos = [G.STYC_RACE_MARKS[m.casefold()] for m in marks]
    lats = np.array([p.lat for p in pos])
    lons = np.array([p.lon for p in pos])
    marks = np.vstack(G.MAP(lons, lats)).T 
    # Add red x's to the chart ABOVE, at the location of the marks
    ch.ax.scatter(marks[:, 0], marks[:, 1], color='red', marker='x')

def save_chart(ch, directory=""):
    "Save a current chart to a file, using time as filename."
    filename = f"current_{utils.time_to_string(ch.datetime)}.pdf"
    path = os.path.join(directory, filename)
    ch.fig.savefig(path, orientation='portrait')

def create_charts(date, start_time, end_time, region, marks=None):
    """
    Create a set of current charts from the UW Live Ocean data, for a given day that span
    from the start time to the finish.

    Times are local times, 24 hours.

    Models are generated daily, and include predicitions for for the next 2 days. If looking
    for predictions on a future date, you must fetch the most recent model.
    """

    if fetch_live_ocean_model(date):
        uw_model = read_live_ocean_model(date)
        time_indices, times  = find_times(uw_model, f"{date} {start_time}", f"{date} {end_time}")
        ch_list = [display_currents(uw_model, region, t) for t in time_indices]
        for ch in ch_list:
            if marks is not None:
                plot_marks(ch, marks)
        return ch_list
    else:
        display("Wait for model file to fetch!")
        return None


def region_from_df(df, lat_border=0.2, lon_border=0.3):
    """
    Extract the geographic region which covers the entire race track.  BORDER is an
    additional margin which ensures you do not bump up against the edge when graphing.
    """
    # Add just a bit of "fudge factor" to ensure that the extent is not too small, which
    # triggers some corner cases.
    fudge = (0.015, -0.015)

    # TODO: since the border is applied in lat/lon separately, its is not uniform.  Same
    # for FUDGE.
    lat_max, lat_min = chart.max_min_with_border(df.latitude, lat_border) + fudge
    lon_max, lon_min = chart.max_min_with_border(df.longitude, lon_border) + fudge

    return DictClass(lat_max=lat_max, lat_min=lat_min,
                     lon_max=lon_max, lon_min=lon_min)


#### Cell #5 Type: module ######################################################

def show_boat_currents(ch, df, dt_seconds=5, scale=1000, leeway=8, multiplier=1.1):
    delay = 16
    dt = dt_seconds * G.SAMPLES_PER_SECOND

    ch_begin = (ch.datetime + datetime.timedelta(hours=-1)).datetime
    ch_end = (ch.datetime + datetime.timedelta(hours=+1)).datetime

    tdf = df[(df.row_times > ch_begin) &  (df.row_times < ch_end)]
    chart.draw_track(df, ch, color='lightgrey')
    chart.draw_track(tdf, ch, color='olive')    

    mdf = tdf.iloc[:-delay:dt]
    ddf = tdf.iloc[delay::dt]

    vog_n = ddf.sog * p.north_d(ddf.cog)
    vog_e = ddf.sog * p.east_d(ddf.cog)

    thdg = mdf.hdg.copy()

    port_hauled = land(mdf.awa < 0, mdf.awa > -120)
    stbd_hauled = land(mdf.awa > 0, mdf.awa < 120)

    thdg[port_hauled] = thdg[port_hauled] + leeway
    thdg[stbd_hauled] = thdg[stbd_hauled] - leeway    
    
    hdg = thdg + df.variation.mean()

    btv_n = multiplier * mdf.spd * p.north_d(hdg)
    btv_e = multiplier * mdf.spd * p.east_d(hdg)

    cur_n = (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = (np.asarray(vog_e) - np.asarray(btv_e))

    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    east, north = G.MAP(longitudes, latitudes)

    ch.ax.quiver(east, north, scale/10 * btv_e, scale/10 * btv_n,
                angles='xy', scale_units='xy', scale=1, color='orange',
                width=0.003)
    
    ch.ax.quiver(east, north, scale/10 * vog_e, scale/10 * vog_n,
                angles='xy', scale_units='xy', scale=1, color='aqua',
                width=0.003)

    
    ch.ax.quiver(east, north, scale * cur_e, scale * cur_n,
                angles='xy', scale_units='xy', scale=1, color='red',
                width=0.003)
        
    # ch.ax.quiver(east, north, scale * btv_e, scale * btv_n,
    #              angles='xy', scale_units='xy', scale=1, color='red')


#### Cell #9 Type: module ######################################################

# This a function cribbed from link sort of verbose but handy.
# http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html#code

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

#### Cell #10 Type: module #####################################################

def to_datetime(seconds):
    "Convenience to convert ncdf times to datetimes."
    epoch = datetime.datetime.utcfromtimestamp(0)
    dt = epoch + datetime.timedelta(0, seconds)
    return arrow.get(dt).to('US/Pacific')

#### Cell #19 Type: module #####################################################

# Compute an image, in our projected coordinates, where the value of the pixel is 
# determined by mapping the lat/lon coordinate into pixel coordinates.
#
# Resulting image is 2*size x 2*size.  Do not process points which are more than 
# threshold distance away in east or north.
#
# For any given image, there are likely to be holes.

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

#### Cell #28 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "timestamp": "2020-06-12T22:14:23.672069-07:00"
#:   }
#: }

#### Cell #29 Type: finish #####################################################

