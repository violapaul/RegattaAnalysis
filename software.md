
# Produce a high quality video with overlay

This code has something very similar to what we want, except for drones!

Flight Data Manager

    https://sites.google.com/site/pud2gpxkmlcsv/

Detailed multi-step directions

    https://www.apollomaniacs.com/ipod/howto_ar_drone_flight_v03_video_gps_en.htm


- Custom widgets (instruments) and custom template (pattern of widgets).
- Conversion of drone data (PUD file?) into a Fit file.

If you decide to create your own template you need the following info (but if you
just use one of the two Bebop templates you can just ignore the following):

In order to make it independently of the units selected in the Garmin Virb settings
menu, speed, altitude and distance is not reported as speed, altitude and distance in
the Garmin Fit file, which would give a lot of problems if the user have made a wrong
setting.

Instead the various data is reported the following way in the Garmin Fit file:

- 'Speed' is reported as 'Stance Time' in Fit file (is found in Garmin Virb under Gauges -> All Running)
- 'Altitude' for text is reported as 'Stance Time Percent' in Fit file (is found in Garmin Virb under Gauges -> All Running)
- 'Altitude over ground' is reported as "Engine Fuel Rate" in Fit file (is found in Garmin Virb under Gauges -> All OBD-II Sensors)
- 'Altitude' for graph is reported 'Elevation' in Fit file (is found in Garmin Virb under Gauges -> All Position)
- 'Drone distance' is reported as 'Engine RPM' in Fit file (is found in Garmin Virb under Gauges -> All OBD-II Sensors)
- 'Distance flown' is reported as 'Power' in Fit file (is found in Garmin Virb under Gauges -> All Sensors)
- 'Battery %' is reported as 'Cadance' in Fit file (is found in Garmin Virb under Gauges -> All Sensors)
- 'Orientation of drone' is reported as 'Aviation Track Angle' in Fit file (is found in Garmin Virb under Gauges -> All Aviation)
- 'Pitch' and 'Roll' is reported as 'Pitch' and 'Roll' in Fit file (is found in Garmin Virb under Gauges -> All Aviation)
- 'WiFi signal level' (only for Bebop2) is reported as 'HeartRate' in Fit file (is found in Garmin Virb under Gauges -> Sensors -> Heart Rate)
- 'Number of GPS satellites' (only for Bebop2) is reported as 'verticalOscillation' in Fit file (is found in Garmin Virb under Gauges -> Running -> Vertical Oscilation)



# How to make after race/sail data preparation easier


# Tenets
- Never delete data permanently
- Keep the raw data, there is no telling if a bug in the pipeline 
- Process all data, but flag useful vs useless.
- Only copy changes, to make things fast
- It should be easy to navigate data
  - Filenames should make sense and be organized
  - Additional DBs that cannot be easily hand editted are bad.  YAML?
- RPi is ephemeral.  Assume it can be lost!

# TODO
- Eliminate full rsync.  Might be expensive, and could compete with logging.  Not
    needed.
- Add a full_backup script that does do a full backup!  Rsync, etc.
- Delete old logs from RPi.
- Reprocess all old logs.

# User Experience
- Come home, pull out USB stick.
- Put in Mac.
- Run pull data script
  - Copies all data over to Mac
  - Finds new logs and processes them.

# Design
- Invariant: All processed logs files are compressed (saves 10x for 300 Megs).
- Operations are idempotent where possible (you can run them again and again).
  - Operations compute what would result, and then only trigger if needed.
- All data kept in a raw logs directory, using RPi filenames
  - Find the new files on the USB stick which do not have compressed versions.
  - Copy and then compress.  *This deletes the uncompressed version*
- Maintain another directory with links to useful logs, with good names (based on
    datetime and boat name).
- Keep things simple. Operations are: `op file_in file_out` with an optional `force`.
  - If `file_out` exists, then you skip the step unless `force`.
- Operations:
  - `copy_rpi_log_file(
  - `extract_file_name(compressed_log_file)`:  Given a raw log, extract a better filename.
    - Got to be super careful here that we are using the **first** datetime record!
        Otherwise multiple files can be generated.
  - `link_file(compressed_log_file, new_path)`
  - `canboat_to_json(compressed_log_file, json_path)`
  - `json_to_gpx(json_path, gpx_path)`
  - `json_to_pandas(json_path, gpx_path)`
- Orchestration can be done by the file, or by the directory.
  - Process directory: find all files, and then trigger the pipeline.
    - User will want this, because it will automaically process all new logs
  - Or a single file can be processed, most likely with `force`.

- Does a processing step understand 


# Questions
- what processing on the RPi vs. on the laptop?
  - generally want to keep the RPi very simple... for now.
- Some files are in the git repo, others not.  Is this good?
- What about the err files?
- Do we need to clean up files?  Remove from RPi?
  - Mostly empty files, from runs at home where there is no canbus
  - very old files

# Facts
- Current logs go back to about October
- Raw logs are large, up to 350 Meg, (compressed is 1/10 th)
- Raw logs are processed into: GPX files, JSON files, and Pandas files.
  - All are large and could be usefully compressed
  - JSON is not useful in itself, but is more easily readable than raw.
- We are unlikely to collect more than 10 real logs a month
- Logging takes about 1Meg a minute.  Less than 10Meg is not useful.


# Annoying facts
- The RPi does not have a battery backed clock, so it does not know the real
    datetime.
  - The log files created cannot have the real date time as a name, though it is
      handy.
  - The real datetime is available in the file itself.

# Ideas
- Does it make sense to use symlinks instead of renaming of files
- Or just skip renaming, use the RPi log files, and use a association table.
  - I have found this disorienting, since the filenames are pretty much random.



 Required steps.

# Get the data off the RPi.
 - Has the data already been moved?
 - Where does it go
- Figure out what data is new.
- Process that data.
Discard 


# List of stuff

Directory where I am playing with GDAL.  Check out the readme
    
    Sailboat/Code/TestData/GDAL/




# TODO

- Smooth the polars??
- Compare the polars with the measured data.
- Organize the data by date and time.
- Deal with the fact that we are not always racing...  and 
- Plot the data on a map


# Polars 

## The data

Been playing around with various polars.

I have a very old VPP/ORC report for the J105.  This has graphical figures and
tables.  I have converted the tables to text and loaded them into python, and then
plotted this data.

One issue is that the graphs seem to have a lot more data in them than the tables...
in particular the tables clearly show TWS/TWA vs boat speed, but not for the higher
angles.  These are represented by the best VMG for each TWS.  In order to get a full
curve you need to interpolate and extrapolate.

I am not entirley happy with the "smoothness" of these graphs.

After some search settled on RBFs with quintic BFs.  Reasonably smooth and reasonably
accurate.  Max error is at the cusp between the jib/kite transition (where there is a
small cusp).  Low errors upwind and downwind (low at max VMG).

```
smooth=100000, 
function='quintic'
epsilon = None  # ignored for quintic
rbf_spline = scipy.interpolate.Rbf(defined.tws, defined.twa, defined.btv, smooth=smooth, function=function, epsilon=epsilon)
```

There is also a ton of info on the web (but not great



# Python goodness

See source code

    import inspect
    source_DF = inspect.getsource(pandas.DataFrame)

Reload module (careful to be in the correct directory/path)

    import importlib
    importlib.reload(cb)  # use alias


# Signal-k

My core interest is to log all the data on the boat.  I can then analyze it at
another time.

Ultimately it would be cool to provide a realtime polar chart and feedback.  Or to
build a start line system.  Or to build my own chart plotter.  But this is for later.
The fancy SIGNAL-K servers potentially support this (as plugins).  But they are way
complex, and hard to navigate.  And I hate web programming.  Hurts my head.

Signal-k is a very large project that seems overly complex.  At its core it is a
wrapper on CANBoat (which reads NMEA2000 data over USB from the Actisense).  Canboat
can either capture the raw NMEA2K data (as a stream of records) or decode it into its
own sort of bespoke schema (JSON is the best option here).  Signal-k goes further
still to define a bunch of general taxonomy for boat info, and then convert the
canboat data to signal-k data.  In theory one could then further serialize this data
and receive it on signal-k compatible hardware (as opposed to NMEA2k compatible
hardware).

I have somehow fallen into the trap of using the node based Javascript
implementation, which is a living hell.  The JS version also has its own version of
canboat-js.

I have also tried the Java system, which has its own amazing complexity (including
running influxDB locally).

So my current plan is to strip it all away and just run Canboat as a service which
will read and log all onboard data (see below).  

# CanBoat

## Processing canboat data

### Redundant info

Let's throw out the GPS messages from the Zeus display (src=5).  These are redundant
with the ZG100 unit (src=127), and I have reason to believe that the ZG100 is better.

```
[~/canlogs/JSON]$ head -2000 * | grep "src..5" | grep -iv ais | sed 's/^.*\(src.*fields\).*$/\1/p' | sort | uniq
src":5,"dst":255,"pgn":126992,"description":"System Time","fields
src":5,"dst":255,"pgn":127258,"description":"Magnetic Variation","fields
src":5,"dst":255,"pgn":129025,"description":"Position, Rapid Update","fields
src":5,"dst":255,"pgn":129026,"description":"COG & SOG, Rapid Update","fields
src":5,"dst":255,"pgn":129029,"description":"GNSS Position Data","fields
src":5,"dst":255,"pgn":129539,"description":"GNSS DOPs","fields
src":5,"dst":255,"pgn":129540,"description":"GNSS Sats in View","fields
[~/canlogs/JSON]$ head -2000 * | grep "src..127" | grep -iv ais | sed 's/^.*\(src.*fields\).*$/\1/p' | sort | uniq
src":127,"dst":255,"pgn":126992,"description":"System Time","fields
src":127,"dst":255,"pgn":127258,"description":"Magnetic Variation","fields
src":127,"dst":255,"pgn":129025,"description":"Position, Rapid Update","fields
src":127,"dst":255,"pgn":129026,"description":"COG & SOG, Rapid Update","fields
src":127,"dst":255,"pgn":129029,"description":"GNSS Position Data","fields
src":127,"dst":255,"pgn":129539,"description":"GNSS DOPs","fields
src":127,"dst":255,"pgn":129540,"description":"GNSS Sats in View","fields
```

## Peer Gynt Specific stuff

I suppose the one advantage of Signal-k is that it may normalize the messages that
are sent, into documented schema.  The disadvantage of canboat is that each boat is
potentially different.  

For example on Peer Gynt there are two sources of GNSS data, src=5 and src=127.  They
both seem send the same info, though 127

## Info

This is what I really need for now.  It reads the data and logs it.

    https://github.com/canboat/canboat

Two pieces: 

1) actisense-serial: reads from the Actisense (reverse engineered I beleive). It
creates data records that looks like this:

    2011-11-24-22:42:04.388,2,127251,36,255,8,7d,0b,7d,02,00,ff,ff,ff
    2011-11-24-22:42:04.390,2,127250,36,255,8,00,5a,7c,00,00,00,00,fd
    2011-11-24-22:42:04.437,2,130306,36,255,8,b1,5c,00,ee,f0,fa,ff,ff

These are not decoded...  they are in raw NMEA which is only defined using the NMEA
standards (which is proprietary).

2) analyzer

    $ analyzer < small.log
    N2K packet analyzer $Rev: 233 $ from $Date: 2011-11-27 22:21:08 +0100 (zo, 27 nov 2011) $
    (C) 2009-2011 Keversoft B.V., Harlingen, The Netherlands
    http://yachtelectronics.blogspot.com

    New PGN 127251 for device 36 (heap 5452 bytes)
    2011-11-24-22:42:04.388 2  36 255 127251 Rate of Turn:  SID = 125; Rate = 0.0934 deg/s
    New PGN 127250 for device 36 (heap 5467 bytes)
    2011-11-24-22:42:04.390 2  36 255 127250 Vessel Heading:  SID = 0; Heading = 182.4 deg; Deviation = 0.0 deg; Variation = 0.0 deg; Reference = Magnetic
    New PGN 130306 for device 36 (heap 5480 bytes)
    2011-11-24-22:42:04.437 2  36 255 130306 Wind Data:  SID = 177; Wind Speed = 0.92 m/s; Wind Angle = 353.4 deg; Reference = Apparent

Can also be generated in JSON.

    {"timestamp":"2011-11-24-22:42:04.388","prio":"2","src":"36","dst":"255","pgn":"127251","description":"Rate of Turn","fields":{"SID":"125","Rate":"0.0934"}}
    {"timestamp":"2011-11-24-22:42:04.390","prio":"2","src":"36","dst":"255","pgn":"127250","description":"Vessel Heading","fields":{"SID":"0","Heading":"182.4","Deviation":"0.0","Variation":"0.0","Reference":"Magnetic"}}

# Setting up a service

Helpful info:  https://www.shellhacks.com/systemd-service-file-example/

I wrote a bash script that does the boring job of repeated trying to open the
actisense-serial program and redirecting OUT and ERR.  Its in a loop, so if anything
simple goes wrong it should just restart.

# The Overall Goal is to handle GEO data

Their are multiple sources of GEO data that I would like to integrate.

- The boat itself (location and how that relates to other measurements).
- Weather data
- Perhaps current and tide data.
  - Can we convert the tide map to something quantitative?  Scan images, use this as
    a prior.  Collect data while sailing and try to measure.
- Map data

As a first cut I would like to build a simple visualizer that would show a race/sail
along with various related pieces of info.

- The tacks and how we did.
- The polars
- Speed, groove?

# The lay of the land

Code is in Sailboat/Code

# Tasks

- Figure out how to display maps (how about just the GRIB weather data).
  - With axes correct?
  - Basemap?

# Software packages

## GDAL 

    https://gdal.org/index.html

GDAL is a translator library for raster and vector geospatial data formats that is
released under an X/MIT style Open Source License by the Open Source Geospatial
Foundation. As a library, it presents a single raster abstract data model and single
vector abstract data model to the calling application for all supported formats. It
also comes with a variety of useful command line utilities for data translation and
processing. 

SYSTEM: I have standardized on a single version of Python 3.6.  The GDAL that is
pre-built uses another (I think).  I built GDAL from source (in Packages).  And
installed.  It magically went into the right place.  Followed the directions here:
https://pypi.org/project/GDAL/

### Really good intro to GDAL




## LibBSB

Leagacy file format for NOAA charts (referred to as RNC - raster navigational charts).

SYSTEM: see source in Packages/libbsb-0.0.7 , simple configure and then make (old
school).

Note, there is some metadata, so this might be a better way to import a map.  But it
is ultimately too tightly tied to the chart, which makes it hard to use (it is
literally a scan of the chart).


# Types of data


## GRIB

GRIB is a format for gridded data (exclusively for points on the surface of the
earth, I think).  Data is scalar, with associated Lat/Lon coordinate.  If you want a
vector than it appears to be sent as two different GRIB records (wind direction).

# NOAA tides, etc.

    https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml?#declination

Cool declination page.


## NOAA Charts, PDF, RNC/BSB, and ENC

General overview is [here](https://oceanservice.noaa.gov/facts/find-charts.html).

PDF, is what it is.  Scanned charts.  Not necessarily where you want them, and hard
to use digitally.

NOAA RNC, BSB (Raster Navigational Charts). These are just the PSF but in a NOAA
approved format (before PDF?).  Note BSB is an image format that can be converted to
TIFF with LibBSB and GDAL (see above).

    https://en.wikipedia.org/wiki/Raster_Navigational_Charts_(NOAA)


ENC - Electronic Navigational Charts (vector based and in layers).

The NOAA ENC data are in International Hydrographic Organization (IHO) S-57 format,
which is the data standard for the exchange of digital hydrographic data. Nautical
chart features contained within a NOAA ENC provide a detailed representation of the
U.S. coastal and marine environment. This data includes coastal topography,
bathymetry, landmarks, geographic place names, and marine boundaries. Features in a
NOAA ENC are limited in that they only represent the geographic region depicted in
that particular ENC. Aggregating nautical features from all NOAA ENCs in the creation
of GIS data results in a contiguous depiction of the U.S. coastal and marine
environment.

    http://iho.int/iho_pubs/maint/S57md8.pdf

Chart tiles.  These are available in multiple formats (web streaming based), but
perhaps the easiest is a cached, large (450M), offline collection of all tiles for
the seattle region (Oregon to Canada).  The file is MBTiles format (a single SQLLite
file).  This file can be read by GDAL.  And extracted.

    https://tileservice.charts.noaa.gov/tileset.html#50000_1-locator
    https://tileservice.charts.noaa.gov/mbtiles/50000_1/MBTILES_06.mbtiles



# Starting with downloading weather data: GRIB

NOAA has tons of data.  One of the most interesting is the HRRR data in GRIB.

https://nomads.ncep.noaa.gov/

## Fast Downloading w/ Partial http transfers

https://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html

## GRIB subsets

(Note, this does not quite work.)

And under that: https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?dir=%2Fhrrr.20190920%2Fconus

This provides something of a UX to select a subset of the HRRR data.  This seems to
be a bit finicky sometimes you get what you want, other times not.

And this is a very specific URL generating a single smallish GRIB file:

    `https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t00z.wrfsfcf00.grib2&lev_10_m_above_ground=on&var_GUST=on&var_WIND=on&subregion=&leftlon=220&rightlon=250&toplat=50&bottomlat=40&dir=%2Fhrrr.20190920%2Fconus`

Only returns 1 variable (not two).  There seems to be magical subsets which are pre-computed (and therefore supported).

The one below seems to work "best" returning wind, wind direction (U and V fields),
and what appears to be MAX values for the U and V fields (though not certain).  Note,
I am requesting **all** vars, but getting only 5.  This may be the "standard" thing
to do for wind.

    https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t00z.wrfsfcf05.grib2&
        lev_10_m_above_ground=on&
        all_var=on&
        subregion=&
        leftlon=220&
        rightlon=250&
        toplat=40&
        bottomlat=50&
        dir=%2Fhrrr.20190920%2Fconus

https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t00z.wrfsfcf05.grib2&lev_10_m_above_ground=on&all_var=on&subregion=&leftlon=235&rightlon=245&toplat=49&bottomlat=45&dir=%2Fhrrr.20190920%2Fconus


# Reading GRIB files

## Unhelpful docs from ECMWF

    https://apps.ecmwf.int/codes/grib/format/grib1/overview

And

    https://apps.ecmwf.int/codes/grib/format/grib2/overview

## GRIB libraries

Most of the libraries that read GRIB have many annoying dependencies.

### Pupygrib

This one looks pretty clean (but unfortunately not fully working!).

    https://gitlab.com/gorilladev/pupygrib

### Pygrib

Seems to have a bunch of dependencies, but it seems to work *and* it installed with
pip (inspite of dependencies).

    https://jswhit.github.io/pygrib/docs/

Still some missing info on how this all works.








