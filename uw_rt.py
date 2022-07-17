
#### Cell #0 Type: markdown ####################################################

#: # UW Live Ocean Data
#: 
#: Found a potentially great resource at UW:
#: 
#: http://faculty.washington.edu/pmacc/LO/LiveOcean.html
#: 
#: > LiveOcean works a lot like the weather forecast models that we all rely on every day. It takes in information about the state of the ocean, atmosphere and rivers on a given day, and then uses the laws of physics (and a large computer) to predict how the state of the ocean in our region will change over the next few days. The things that the model predicts are currents, salinity, temperature, chemical concentrations of nitrate, oxygen, carbon, and biological fields like phytoplankton, zooplankton, and organic particles. It does this in three dimensions, and allowing continuous variation over the full 72 hour forecast it makes every day.
#: 
#: - [How the Model Works](http://faculty.washington.edu/pmacc/LO/how_it_works.html)
#: 
#: > The model framework we use is called the Regional Ocean Modeling System (ROMS) which is a large, flexible computer code that is used by coastal and estuarine oceanographers around the world to simulate different regions. It is known to have excellent numerical properties, such as low numerical diffusion, many choices for turbulence parameterization and open boundary conditions, built-in biogeochemical models, and a large, active user community. We have been using ROMS for realistic simulations in the region for research projects for the past decade and have found it to be a robust tool.
#: 
#: Its based on the [ROMS Finite Element Model](https://www.myroms.org/)
#: 
#: - [Awesome Accessible Description of Tides](http://faculty.washington.edu/pmacc/LO/tides_background.html)
#: 
#: ## Data From Parker MacCready
#: 
#: Exchanged emails with Prof. MacCready and he pointed me toward his daily model outputs.


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

from latlonalt import LatLonAlt as lla

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

def fetch_live_ocean_model(date_string, background=True, speed='1000k'):
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
        # command = f"(wget {url} -O {download_path} -c -T 10 --limit-rate={speed} -o {log_file}; mv {download_path} {path})"
        command = f"(curl {url} -o {download_path}; mv {download_path} {path})"
        if background:
            command += "&"
        print(command)
        display(f"Downloading model to {path}.  Check {log_file}.  Incomplete results here: {download_path}.")
        utils.run_system_command(command)
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

def find_times(uw_model, tstart, tfinish, extend=True):
    time_indices = []
    for i, tstamp in enumerate(uw_model.ocean_time):
        t = utils.time_from_timestamp(tstamp)
        if utils.time_after(tstart, t) and utils.time_after(t, tfinish):
            time_indices.append(i)
    if len(time_indices) == 0:
        return time_indices
    if extend and time_indices[0] > 0:
        time_indices = [time_indices[0]-1] + time_indices
    if extend and time_indices[-1] < len(uw_model.ocean_time)-1:
        time_indices = time_indices + [time_indices[-1]+1]
    return time_indices, [utils.time_from_timestamp(uw_model.ocean_time[t]) for t in time_indices]


def mark_positions(marks):
    return [G.mark_position(m) for m in marks]

def region_from_marks(marks, lat_border=0.2, lon_border=0.3):
    pos = mark_positions(marks)
    return region_from_latlon(pos, lat_border, lon_border)

def region_from_latlon(pos, lat_border=0.2, lon_border=0.3):
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

def display_currents(uw_model, region, time_index, plot_wind=False):
    """
    Draw a chart overlayed with the UW live ocean current predictions.
    """
    ch = chart.create_chart(region, pixels=2000)
    ch.fig = plt.figure(figsize=(8, 10))
    ch.ax = ch.fig.add_subplot(111)   
    ch = chart.draw_chart(ch, ch.ax)
    return draw_current(ch, uw_model, time_index, plot_wind=plot_wind)


def draw_current(ch, uw_model, time_index, location=(2700, 737), plot_wind=False):


    # How long should a current vector be?  Given the scale of the map (which may be 10's
    # of Ks), you want the vectors to be visible.  Current is stored in meters/sec. And we
    # typically think in knots.  
    
    scale = 500 * G.MS_2_KNOTS  # a 1 kt current will be 500m long
    u = scale * uw_model.current_e[time_index, :, :]
    v = scale * uw_model.current_n[time_index, :, :]

    wscale = 100
    wu = wscale * uw_model.wind_e[time_index, :, :]
    wv = wscale * uw_model.wind_n[time_index, :, :]
    
    dt = utils.time_from_timestamp(uw_model.ocean_time[time_index])
    ch.datetime = dt

    # Note, for masked arrays mask is TRUE for undefined.
    lat_mask = lor(uw_model.lat_rho > ch.lat_max, uw_model.lat_rho < ch.lat_min)
    lon_mask = lor(uw_model.lon_rho > ch.lon_max, uw_model.lon_rho < ch.lon_min)
    ll_mask = lor(lat_mask, lon_mask)

    mask = np.logical_not(lor(ll_mask, u.mask, v.mask))

    # Draw a set of scale keys that show 1 knot 
    one_knot = (1 / G.MS_2_KNOTS) * scale
    if location is None:
        m_east  = np.mean(uw_model.east[mask])
        m_north = np.mean(uw_model.north[mask])
    else:
        m_east, m_north = location
    ch.ax.arrow(m_east, m_north, one_knot/np.sqrt(2), one_knot/np.sqrt(2),
                head_width=100, length_includes_head=True, color='red')
    ch.ax.arrow(m_east, m_north, one_knot, 0,
                head_width=100, length_includes_head=True, color='red')
    ch.ax.arrow(m_east, m_north, 0, one_knot,
                head_width=100, length_includes_head=True, color='red')

    # Threshlold to display arrow in red.
    theta = (1.0 / G.MS_2_KNOTS) * scale  # above 1.0 knots in red
    m1 = land(mask, np.sqrt(np.square(u) + np.square(v)) >= theta)
    shaft_width = 0.0015
    qscale = 2.0
    ch.ax.quiver(uw_model.east[m1], uw_model.north[m1], u[m1], v[m1], 
                 angles='xy', scale_units='xy', scale=qscale, color='red', width=shaft_width)

    m2 = land(mask, np.sqrt(np.square(u) + np.square(v)) < theta)
    ch.ax.quiver(uw_model.east[m2], uw_model.north[m2], u[m2], v[m2], 
                 angles='xy', scale_units='xy', scale=qscale, color='blue', width=shaft_width)

    if plot_wind:
        shaft_width = 0.002
        ch.ax.quiver(uw_model.east[mask], uw_model.north[mask], wu[mask], wv[mask], 
                     angles='xy', scale_units='xy', scale=1, color='limegreen', width=shaft_width)

    ch.ax.set_title(uw_model.date + " : " + utils.time_to_string(dt))
    return ch

def plot_marks(ch, marks):
    mark_positions = [G.mark_position(m) for m in marks]
    plot_positions(ch, mark_positions)

def plot_positions(ch, positions):
    "Plot the STYC marks identified by name."
    lats = np.array([p.lat for p in positions])
    lons = np.array([p.lon for p in positions])
    marks = np.vstack(G.MAP(lons, lats)).T 
    # Add red x's to the chart ABOVE, at the location of the marks
    ch.ax.scatter(marks[:, 0], marks[:, 1], color='green', marker='x')

def save_chart(ch, directory="", filetype="pdf"):
    "Save a current chart to a file, using time as filename."
    filename = f"current_{utils.time_to_string(ch.datetime)}.{filetype}"
    path = os.path.join(directory, filename)
    ch.fig.savefig(path, orientation='portrait', dpi=300)

def create_charts(date, start_datetime, finish_datetime, region, marks=None, plot_wind=False, time_extend=True):
    """
    Create a set of current charts from the UW Live Ocean data, for a given day that span
    from the start time to the finish.

    Times are local times, 24 hours.

    Models are generated daily, and include predicitions for for the next 2 days. If looking
    for predictions on a future date, you must fetch the most recent model.
    """

    if fetch_live_ocean_model(date):
        uw_model = read_live_ocean_model(date)
        time_indices, times  = find_times(uw_model, start_datetime, finish_datetime, extend=time_extend)
        ch_list = [display_currents(uw_model, region, t, plot_wind=plot_wind) for t in time_indices]
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

def show_boat_currents(ch, df, dt_seconds=60, scale=1000, leeway=5, compass_offset=0, multiplier=1.1):
    delay = 16    # samples
    dt = dt_seconds * G.SAMPLES_PER_SECOND

    ch_begin = (ch.datetime + datetime.timedelta(hours=-1)).datetime
    ch_end = (ch.datetime + datetime.timedelta(hours=+1)).datetime

    tdf = df[(df.row_times > ch_begin) & (df.row_times < ch_end)]
    chart.draw_track(df, ch, color='lightgrey')
    chart.draw_track(tdf, ch, color='olive')    

    mdf = tdf.iloc[:-delay:dt]
    ddf = tdf.iloc[delay::dt]

    vog_n = ddf.sog * p.north_d(ddf.cog)
    vog_e = ddf.sog * p.east_d(ddf.cog)

    thdg = mdf.hdg.copy()

    port_hauled = land(mdf.awa < 0, mdf.awa > -120)
    stbd_hauled = land(mdf.awa > 0, mdf.awa < 120)

    thdg[port_hauled] = compass_offset + thdg[port_hauled] + leeway 
    thdg[stbd_hauled] = compass_offset + thdg[stbd_hauled] - leeway    
    
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


#### Cell #6 Type: notebook ####################################################

# notebook - plot charts for a day on the water

FOULWEATHER_BLUFF = dict(
    start = lla.from_degrees_minutes((47, 48.594), (-122, 28.371)),
    fwb = lla.from_degrees_minutes((47, 56.850), (-122, 36.075)),
    scatchet_head = lla(47.908261, -122.43814),
    pilot_point = lla.from_degrees_minutes((47, 52.8), (-122, 30.5)))

RACE_WEEK = dict(
    a = lla(48.558397865660154, -122.59793899817099),
    b = lla(48.7153969326876, -122.71810196439795),
    c = lla(48.71856817245304, -122.51897475846975))
            
SAN_JUANS = dict(
    SW = lla(48.192903, -123.199949),
    N = lla(48.769520, -122.930888),
    SE = lla(48.136633, -122.639886)
)

SCATCHET_HEAD = dict(
    sh = lla(47.908261, -122.43814),
    j = G.STYC_MARKS['j'],
    n = G.STYC_MARKS['n']
    )

POSSESSION_PT = dict(
    sh = lla.from_degrees_minutes((47, 54.070), (-122, 23.012)),
    j = G.STYC_MARKS['j'],
    n = G.STYC_MARKS['n']
    )

WINTER_VASHON = dict(
    nw = lla.from_degrees_minutes((47, 31.033), (-122, 29.154)),
    east = lla.from_degrees_minutes((47, 23.147), (-122, 20.951)),
    south = lla.from_degrees_minutes((47, 19.122), (-122, 33.334))
)

ROUND_THE_SOUND = dict(
    n = G.STYC_MARKS['n'],
    nw = lla.from_degrees_minutes((47, 31.033), (-122, 29.154)),
    east = lla.from_degrees_minutes((47, 23.147), (-122, 20.951)),
    south = lla.from_degrees_minutes((47, 19.122), (-122, 33.334)),
    pt_townsend = lla(48.129727, -122.748464),
    hat_island = lla(48.025588, -122.329610)
    )

THREE_TREE_POINT = dict(
    n = G.STYC_MARKS['n'],
    s = lla.from_degrees_minutes((47, 39.720), (-122, 29.790)),
    d = G.STYC_MARKS['d'],
    opposite_ttp = lla(47.452433868146834, -122.43496184038126)
)

JIM_DEPUE = [G.STYC_MARKS[m] for m in 'kwjom']
BLAKELY_ROCKS = [G.STYC_MARKS[m] for m in 'nku']
THREE_BUOY = [G.STYC_MARKS[m] for m in 'nbzjr']
JFEST = [G.STYC_MARKS[m] for m in 'mbn']
FALL_REGATTA = [G.STYC_MARKS[m] for m in 'rwms']

if True:
    date = "2022-07-09"
    fetch_live_ocean_model(date)

if True:
    plt.close('all')
    # positions = [ROUND_THE_SOUND[k] for k in "n nw east south".split()]
    # positions = [ROUND_THE_SOUND[k] for k in "n pt_townsend hat_island".split()]
    # positions = list(FOULWEATHER_BLUFF.values())
    positions = FALL_REGATTA

    region = region_from_latlon(positions, 0.2, 0.3)

    sdate = "2021-10-16"
    tstart = utils.time_from_string(sdate + " 10:00")
    tfinish = utils.time_from_string(sdate + " 17:00")

    ch_list = create_charts(date, tstart, tfinish, region, plot_wind=False)
    if ch_list is not None:
        chart_dir = "/Users/viola/tmp"
        display(f"Saving charts in {chart_dir}")
        for ch in ch_list:
            plot_positions(ch, positions)
            save_chart(ch, chart_dir, "jpg")

            
df, race = race_logs.read_date(sail_date)
if True:
    plt.close('all')
    start_time = "10:55:00"
    end_time = "11:05:00"
    ch_list = create_charts(date, sail_date, start_time, end_time, region, plot_wind=False, time_extend=False)
    ch = ch_list[0]
    race_chart = chart.draw_track(df, ch)
    show_boat_currents(ch, df, leeway=5, multiplier=1.15, compass_offset=-8)

    
if True:
    ch_begin = (ch.datetime + datetime.timedelta(hours=-1)).datetime
    ch_end = (ch.datetime + datetime.timedelta(hours=+1)).datetime

    tdf = df[(df.row_times > ch_begin) & (df.row_times < ch_end)]

    chart.quick_plot(df.index, (df.spd, df.sog), legend="spd sog".split())
    chart.chart_and_plot(df, df.index, (df.spd, df.sog))

    params = dict(
        gps_delay = 16,
        dt = 60 * G.SAMPLES_PER_SECOND,
        dt_seconds=60,
        scale=1000,
        leeway=5,
        compass_offset=0,
        multiplier=1.1,
        variation = df.variation.mean(),
    )

    trimmed_df = tdf.iloc[:-params.gps_delay : params.dt]
    ddf = tdf.iloc[params.gps_delay : : params.dt]

    vog_n = trimmed_df.sog * p.north_d(trimmed_df.cog)
    vog_e = trimmed_df.sog * p.east_d(trimmed_df.cog)

    port_hauled = land(trimmed_df.awa < 0, trimmed_df.awa > -120)
    stbd_hauled = land(trimmed_df.awa > 0, trimmed_df.awa < 120)

    thdg = trimmed_df.hdg.copy()
    thdg[port_hauled] = params.compass_offset + thdg[port_hauled] + params.leeway 
    thdg[stbd_hauled] = params.compass_offset + thdg[stbd_hauled] - params.leeway    
    
    hdg = thdg + params.variation

    btv_n = params.multiplier * ddf.spd * p.north_d(hdg)
    btv_e = params.multiplier * ddf.spd * p.east_d(hdg)

    cur_n = (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = (np.asarray(vog_e) - np.asarray(btv_e))

    

    

print(True)
    
#### Cell #8 Type: notebook ####################################################

# notebook 

# I was able to grab an NC file from:
# https://pm2.blob.core.windows.net/f20200520/ocean_surface.nc
#
# From: Parker MacCready <p.maccready@gmail.com>
# It has the surface currents (u,v) for all 73 hours of today's forecast on the 
# lon_rho, lat_rho grid.  New ones appear every day by around 8 AM.

data_dir = "/Users/viola/BigData"
model_file = "2020-06-01_ocean_surface.nc"

ncdf_ocean = netCDF4.Dataset(os.path.join(data_dir, model_file))

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

#### Cell #11 Type: notebook ###################################################

# notebook

# Let's see what is in the file (as advertized it is very similar to ncdump)
ncdump(ncdf_ocean)

#### Cell #12 Type: markdown ###################################################

#: ### NetCDF is self documenting.
#: 
#: We are specifically interested in U,V which are the current predictions in m/s (note, it does not say current, but Parker's email points this out).  
#: 
#: We are likely also interested in Uwind and Vwind (if they turn out to be more accurate than other local predictions).  **But are these inputs to the ocean model, or outputs?** And if inputs, from what model?
#: 
#: NetCDF is a gridded representation.  In this case the "data" is triple indexed
#: by `('ocean_time', 'eta_rho', 'xi_rho')`.
#: 
#: But what does this mean?  The data is on a fixed grid indexed by integers, but the **meaning** of that grid location is held in the associated variables.

#### Cell #13 Type: notebook ###################################################

# notebook

display_markdown("## Meaning of the index variables.")

for name in "ocean_time lon_rho lat_rho".split():
    display_markdown(f"### {name}")
    display(ncdf_ocean.variables[name])
    display(ncdf_ocean.variables[name][:])

ocean_time = ncdf_ocean.variables['ocean_time'][:]
lon_rho = ncdf_ocean.variables['lon_rho'][:]
lat_rho = ncdf_ocean.variables['lat_rho'][:]

[to_datetime(ot) for ot in ocean_time]

# Similarly we can extract the actual current data:

current_n = ncdf_ocean.variables['v'][:, :, :]
current_e = ncdf_ocean.variables['u'][:, :, :]
wind_n = ncdf_ocean.variables['Vwind'][:, :, :]
wind_e = ncdf_ocean.variables['Uwind'][:, :, :]

im = current_n[0, :, :]
display(im)
display(im.shape)

#### Cell #14 Type: markdown ###################################################

#: ### Note that the array is masked. 
#: 
#: https://numpy.org/doc/stable/reference/maskedarray.generic.html
#: 
#: And this makes sense, since the value of the current is not defined on land, etc."
#: 
#: Note, `True` implies that the data is **NOT** valid.
#: 
#: And conveniently matplotlib handles masked array directly.

#### Cell #15 Type: notebook ###################################################

# notebook

fig, ax = plt.subplots(1, 1, num=None)
fig.tight_layout()
ax.imshow(im)

#### Cell #16 Type: markdown ###################################################

#: ### What is going on with this?
#: 
#: Looking at the image, it appears to be corrupted.  Its not.  (While it is ultimately not important, you can "see" the map if you flip it vertically and then ignore the distortion.  Seattle is at roughly x=602, y=745.)
#: 
#: Remember that each pixel in the image is the speed of the current at a particular time and location.

#### Cell #17 Type: notebook ###################################################

# notebook

i = 0
j = 745
k = 602
display(f"Northward current is: {current_n[i, j, k]}")  # is the value at 
display(f"time = {to_datetime(ocean_time[i]).isoformat()}")
display(f"longitude = {lon_rho[j, k]}")
display(f"latitude = {lat_rho[j, k]}")

display("Lat/Lon of Seattle is: 47.6062° N, 122.3321° W")

# We can convert these lat/lon coordinate to a local projection using Pyproj.  

east, north = G.MAP(lon_rho, lat_rho)  # Works great with numpy masked arrays.

# The result is in meters east from our center of projection.
east

#### Cell #18 Type: markdown ###################################################

#: ### Visualizing Data
#: 
#: We now have a subtle problem, the data is not really an image.  There are parts of the map where the sampling coarse and others where it is fine (there are more pixels in some places and less others).  We can perform a quick hack to turn this into an image as follows:

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

#### Cell #20 Type: notebook ###################################################

# notebook

dist = 40000
# First construct a higher resolution image
res = forward_map(east, north, im, dist, 200)

plt.figure()
plt.imshow(res, extent=[-dist, dist, -dist, dist])

# Second a lower res image.
res = forward_map(east, north, im, dist, 75)

plt.figure()
plt.imshow(res, extent=[-dist, dist, -dist, dist])


#### Cell #21 Type: markdown ###################################################

#: ### Looks like Seattle, but notice the holes
#: 
#: Specifically zoom in on the higher resolution image.  The Live Ocean data is 0.5km.  When the image resolution is higher than 0.5km then there are pixels where there is no estimate.  Using a lower resolution image (second image) fills in the gaps, but it is unnecessarily coarse.
#: 
#: Note, ultimately these images are not particularly useful.  But it does show the structure of the underlying data.
#: 
#: We can also fill in the holes, using a blend of nearby pixels.  Note, this does a nice job **except** near the coastline, where it blurs things (it is unaware of the coast).

#### Cell #22 Type: notebook ###################################################

# notebook

def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    middle = np.int(kernel_size/2)
    for c in range(kernel_size):
        for r in range(kernel_size):
            d = np.square(c - middle) + np.square(r - middle)
            kernel[c, r] = np.exp(-sigma * d/np.square(middle))
    return kernel

def image_fill(image, kernel_size):
    "Fill in missing values in image by taking a weighted blend of nearby pixels."
    kernel = gaussian_kernel(kernel_size, 1.5)
    res = np.zeros(image.shape)
    mask = np.ones(res.shape, np.bool)
    image_fill_helper(image.data, image.mask, res, mask, kernel)
    return ma.array(data=res, mask=mask)
                    
@jit(nopython=True)
def image_fill_helper(image, mask, res, res_mask, kernel):
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

#### Cell #23 Type: notebook ###################################################

# notebook

res = forward_map(east, north, im, dist, 1000)
filled = image_fill(res, 13)

plt.figure()
plt.imshow(filled, extent=[-dist, dist, -dist, dist])

dist = 15000
scale = 1500

maske = lor(east < -dist, east > dist)
maskn = lor(north < -dist, north > dist)

for time in range(0, 1, 1):
    u = scale * current_e[time, :, :]
    v = scale * current_n[time, :, :]

    mask = np.logical_not(lor(maske, maskn, u.mask, v.mask))
    display(to_datetime(ocean_time[time]).isoformat())

    plt.figure()
    plt.quiver(east[mask], north[mask], u[mask], v[mask], 
               angles='xy', scale_units='xy', scale=1)
    plt.axis('equal')


#### Cell #24 Type: notebook ###################################################

# notebook

def example_race(date="2019-11-16"):
    dfs, races, big_df = race_logs.read_dates([date], race_trim=False)
    display(races)
    return dfs[0]

df = example_race("2020-04-28")
model_file = "2020-06-01_ocean_surface.nc"

ncdf_ocean = netCDF4.Dataset(os.path.join(data_dir, model_file))

current_n = ncdf_ocean.variables['v'][:, :, :]
current_e = ncdf_ocean.variables['u'][:, :, :]
wind_n = ncdf_ocean.variables['Vwind'][:, :, :]
wind_e = ncdf_ocean.variables['Uwind'][:, :, :]

ocean_time = ncdf_ocean.variables['ocean_time'][:]
lon_rho = ncdf_ocean.variables['lon_rho'][:]
lat_rho = ncdf_ocean.variables['lat_rho'][:]

ch = chart.plot_chart(df, border=0.7, color='red')

# Note, for masked arrays mask is TRUE for undefined.
lat_mask = lor(lat_rho > ch.lat_max, lat_rho < ch.lat_min)
lon_mask = lor(lon_rho > ch.lon_max, lon_rho < ch.lon_min)
ll_mask = lor(lat_mask, lon_mask)

mm = np.logical_not(lor(maske, maskn))

region = DictClass(lat_max=lat_rho[mm].max(), lat_min=lat_rho[mm].min(),
                   lon_min=lon_rho[mm].min(), lon_max=lon_rho[mm].max())
region

plt.close('all')

for i in [12, 13, 14]:
    display_currents(ch, i)


#### Cell #25 Type: notebook ###################################################

# notebook 

def show_boat_currents(ch, df, df_slice, dt_seconds=5, scale=1000, leeway=8):
    delay = 16
    dt = dt_seconds * G.SAMPLES_PER_SECOND

    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df.loc[ss]
    ddf = df.loc[dss]
    vog_n = ddf.sog * p.north_d(ddf.cog)
    vog_e = ddf.sog * p.east_d(ddf.cog)

    thdg = mdf.hdg.copy()

    port_hauled = land(mdf.awa < 0, mdf.awa > -50)
    stbd_hauled = land(mdf.awa > 0, mdf.awa < 50)

    thdg[port_hauled] = thdg[port_hauled] + leeway
    thdg[stbd_hauled] = thdg[stbd_hauled] - leeway    
    
    hdg = thdg + df.variation.mean()

    btv_n = mdf.spd * p.north_d(hdg)
    btv_e = mdf.spd * p.east_d(hdg)

    cur_n = (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = (np.asarray(vog_e) - np.asarray(btv_e))

    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    east, north = G.MAP(longitudes, latitudes)

    ch.ax.quiver(east, north, scale * cur_e, scale * cur_n,
                 angles='xy', scale_units='xy', scale=1, color='orange')
        
    # ch.ax.quiver(east, north, scale * btv_e, scale * btv_n,
    #              angles='xy', scale_units='xy', scale=1, color='red')
        

    

def show_boat_arrows(ch, df, df_slice, dt_seconds=5, skip=2, current_scale=1):
    delay = 16
    dt = dt_seconds * G.SAMPLES_PER_SECOND
    scale = dt_seconds
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df.loc[ss]
    ddf = df.loc[dss]
    vog_n = scale * ddf.sog * p.north_d(ddf.cog)
    vog_e = scale * ddf.sog * p.east_d(ddf.cog)

    tw_n = scale * ddf.tws * p.north_d(ddf.twd)
    tw_e = scale * ddf.tws * p.east_d(ddf.twd)

    hdg = mdf.hdg + df.variation.mean()
    btv_n = scale * mdf.spd * p.north_d(hdg)
    btv_e = scale * mdf.spd * p.east_d(hdg)

    cur_n = current_scale * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = current_scale * (np.asarray(vog_e) - np.asarray(btv_e))

    ch.mdf = mdf
    ch.ddf = ddf
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(G.MAP(longitudes, latitudes)).T 

    color = 'blue'
    hwidth = scale/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, skip):
        avog = ch.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, skip):
        atw = ch.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, skip):
        abtv = ch.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, skip):
        acurrent = ch.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    ch.ax.legend([avog, atw, abtv, acurrent],
                    'VOG TWD BTV CURRENT'.split(),
                    loc='best')
        
    return chart

#### Cell #26 Type: notebook ###################################################

# notebook

ch = chart.plot_chart(df, color='red')

ll_mask = lor(lat_rho > ch.lat_max, lat_rho < ch.lat_min)
ll_mask = lor(ll_mask, lon_rho > ch.lon_max, lon_rho < ch.lon_min)


for time in [8, 9, 10]:
    cscale = 1000 # 1000
    u = cscale * current_e[time, :, :]
    v = cscale * current_n[time, :, :]

    wscale = 100 # 100
    uwind = wscale * wind_e[time, :, :]
    vwind = wscale * wind_n[time, :, :]
    
    mask = np.logical_not(lor(ll_mask, maske, maskn, u.mask, v.mask, uwind.mask, vwind.mask))
    adt = arrow.get(to_datetime(ocean_time[time]))
    display(adt.to('US/Pacific'))
    
    ch = chart.create_chart(ch)
    ch.fig = plt.figure(figsize=(8, 10))
    ch.ax = ch.fig.add_subplot(111)   
    ch = chart.draw_chart(ch, ch.ax)
    ch.ax.quiver(east[mask], north[mask], u[mask], v[mask], 
                angles='xy', scale_units='xy', scale=1, color='blue')
    ch.ax.quiver(east[mask], north[mask], uwind[mask], vwind[mask], 
                angles='xy', scale_units='xy', scale=1, color='red')
    ch.ax.set_title(model_file + " : " + str(adt.to('US/Pacific')))
    ch.fig.tight_layout()

    ch.fig.savefig("tide_" + str(adt.to('US/Pacific'))+".pdf", orientation='portrait')

print(np.array((v[mask].max(), v[mask].min()))/scale)
print(scale)

plt.close('all')


time = 20
if True:
    u = scale * current_e[time, :, :]
    v = scale * current_n[time, :, :]

    mask = np.logical_not(lor(maske, maskn, u.mask, v.mask))
    display(to_datetime(ocean_time[time]).isoformat())

    
ch = chart.create_chart(region)
ch.fig = plt.figure(figsize=(8, 8))
ch.ax = ch.fig.add_subplot(111)   
ch = chart.draw_chart(ch, ch.ax)
ch.ax.quiver(east[mask], north[mask], u[mask], v[mask], 
            angles='xy', scale_units='xy', scale=1, color='red')


import itertools as it
[(i, to_datetime(ot)) for i, ot in enumerate(ocean_time)]

#### Cell #27 Type: notebook ###################################################

# notebook

def show_boat_arrows(df, df_slice, dt_seconds=5, skip=2, current_scale=1):
    delay = 16
    dt = dt_seconds * G.SAMPLES_PER_SECOND
    scale = dt_seconds
    ss = slice(df_slice.start, df_slice.stop, dt)
    dss = slice(ss.start+delay, ss.stop+delay, dt)
    
    mdf = df.loc[ss]
    ddf = df.loc[dss]
    vog_n = scale * ddf.sog * p.north_d(ddf.cog)
    vog_e = scale * ddf.sog * p.east_d(ddf.cog)

    tw_n = scale * ddf.tws * p.north_d(ddf.twd)
    tw_e = scale * ddf.tws * p.east_d(ddf.twd)

    hdg = mdf.hdg + df.variation.mean()
    btv_n = scale * mdf.spd * p.north_d(hdg)
    btv_e = scale * mdf.spd * p.east_d(hdg)

    cur_n = current_scale * (np.asarray(vog_n) - np.asarray(btv_n))
    cur_e = current_scale * (np.asarray(vog_e) - np.asarray(btv_e))

    chart = plot_chart(mdf, 3, border=0.0)
    chart.mdf = mdf
    chart.ddf = ddf
    longitudes = np.asarray(mdf.longitude)
    latitudes = np.asarray(mdf.latitude)
    pos = np.vstack(G.MAP(longitudes, latitudes)).T - (chart.west, chart.south)

    color = 'blue'
    hwidth = scale/5
    for (east, north), ve, vn in it.islice(zip(pos, vog_e, vog_n), 0, None, skip):
        avog = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'green'
    for (east, north), ve, vn in it.islice(zip(pos, tw_e, tw_n), 0, None, skip):
        atw = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'red'
    for (east, north), ve, vn in it.islice(zip(pos, btv_e, btv_n), 0, None, skip):
        abtv = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    color = 'orange'
    for (east, north), ve, vn in it.islice(zip(pos, cur_e, cur_n), 0, None, skip):
        acurrent = chart.ax.arrow(east, north, ve, vn, head_width=hwidth, length_includes_head=True, color=color)

    chart.ax.legend([avog, atw, abtv, acurrent],
                    'VOG TWD BTV CURRENT'.split(),
                    loc='best')
        
    return chart


#### Cell #28 Type: metadata ###################################################

#: {
#:   "metadata": {
#:     "kernelspec": {
#:       "display_name": "Python [conda env:sail] *",
#:       "language": "python",
#:       "name": "conda-env-sail-py"
#:     },
#:     "language_info": {
#:       "codemirror_mode": {
#:         "name": "ipython",
#:         "version": 3
#:       },
#:       "file_extension": ".py",
#:       "mimetype": "text/x-python",
#:       "name": "python",
#:       "nbconvert_exporter": "python",
#:       "pygments_lexer": "ipython3",
#:       "version": "3.7.0"
#:     },
#:     "timestamp": "2020-06-12T22:14:23.669709-07:00"
#:   },
#:   "nbformat": 4,
#:   "nbformat_minor": 2
#: }

#### Cell #29 Type: finish #####################################################

