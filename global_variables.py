"""
File contains all the global constants that are related to a specific boat, location,
or NMEA network.


Edit this file to match you boat and location.
"""

import os
import logging
import dateutil
import dateutil.tz

import pandas as pd

from latlonalt import LatLonAlt as lla

import pyproj

def setup_logger(level):
    # create logger
    logger = logging.getLogger('regatta_analysis')
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(funcName)s| %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger

# Setup some global variables ################
class RaceAnalysis:
    "Global variables that are shared by most foreseeable uses of the RaceAnalysis software."
    def __init__(self):
        # Use this logger rather than the global logging functions.
        self.logger = setup_logger(logging.DEBUG)
        
        # The NMEA 2k messages that we are interested in.
        self.PGN_NAME = {
            126992: 'System Time',
            127245: 'Rudder',
            127250: 'Vessel Heading',
            127251: 'Rate of Turn',
            127257: 'Attitude (pitch, roll)',
            127258: 'Magnetic Variation',
            128259: 'Speed',
            128267: 'Water Depth',
            129025: 'Position, Rapid Update',
            129026: 'COG & SOG, Rapid Update',
            129029: 'GNSS Position Data',
            129539: 'GNSS DOPs',
            129540: 'GNSS Sats in View',
            130306: 'Wind Data'
        }

        # Reverse index.
        self.PGN_CODE = {v:k for (k,v) in self.PGN_NAME.items()}
        
        self.PGNS_AIS = {
            129039: "AIS Class B Position Report",
            129809: "AIS Class B static data (msg 24 Part A)",
            129810: "AIS Class B static data (msg 24 Part B)",
            130842: "Simnet: AIS Class B static data (msg 24 Part A)",
        }

        self.PGN_WHITELIST = list(self.PGN_NAME.keys())
        self.PGN_AIS_WHITELIST = list(self.PGNS_AIS.keys())

        self.SAMPLES_PER_SECOND = 10
        self.MS_2_KNOTS = 1.944
        self.METERS_PER_FOOT = 0.3048

        self.DATA_DIRECTORY = 'Data'
        # Mount point for the USB memory stick
        self.USB_LOGS_DIRECTORY = 'Data/USB/logs'
        # Directory where compressed canboat logs are stored
        self.COMPRESSED_LOGS_DIRECTORY = 'Data/CompressedLogs'
        self.NAMED_LOGS_DIRECTORY = 'Data/NamedLogs'
        self.GPX_LOGS_DIRECTORY = 'Data/GPXLogs'
        self.PANDAS_LOGS_DIRECTORY = 'Data/PandasLogs'

        # A small log is generally an artifact.
        self.MIN_LOG_FILE_SIZE = 5000000

        self.MAP_DIRECTORY = 'Data/Maps'
        self.LOG_INFO_PATH = os.path.join(self.DATA_DIRECTORY, 'log_info.pd')

        self.TIMEZONE = dateutil.tz.gettz('UTC')

        self.GSHEET_URL = r"https://docs.google.com/spreadsheets/d/e/2PACX-1vS5g8oeSAMk-CFP-xDi4hu9a23W-iF5SMNjap-Gd78BPWvhA1GGgpDqFkQaEUVD3zoM9Pud1fozuDn8/pub?output=csv"
        self.METADATA_PATH = os.path.join(self.DATA_DIRECTORY, 'metadata.yml')

    def set_logging_level(self, level):
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: level")
            self.logger.setLevel(numeric_level)
        else:
            self.logger.setLevel(level)                

    def init_j105(self):
        "Measurements for a J105."
        self.MAST_HEIGHT = 50 * self.METERS_PER_FOOT  # 50 feet in meters
        self.BOAT_NAME = "PeerGynt"

    def init_peer_gynt(self):
        "Measurements from the J105 called Peer Gynt."
        self.init_j105()
        self.DEVICES = {
            "Actisense": 0,
            "BT-1": 1,
            "Triton Autopilot Keypad": 2,
            "Autopilot": 3,
            "RC42 Rate Compass": 4,
            "Zeus iGPS": 5,
            "Zeus Pilot Controller": 6,
            "Zeus Navigator": 7,
            "V50 Radio": 8,
            "Triton Display 0": 9,
            "Triton Display 1": 10,
            "Triton Display 2": 11,
            "Triton Display 3": 12,
            "Triton Display 4": 13,
            "Wind Sensor": 14,
            "Zeus MFD": 16,
            "Airmar Depth/Speed Transducer DST200": 35,
            "NAIS-400": 43,
            "ZG100 Antenna": 127,
            "ZG100 Compass": 128
        }

    def init_seattle(self, logging_level=logging.INFO):
        "Localize to Seattle."

        self.init_peer_gynt()
        self.set_logging_level(logging_level)

        self.LOCALE = "Seattle"
        # By centering everything here, we can easily compare North/East (NED) coordinates across
        # runs and races, etc.
        self.LATITUDE_CENTER    = 47.687307
        self.LONGITUDE_CENTER = -122.438644
        self.LATLON = lla(lat=self.LATITUDE_CENTER, lon=self.LONGITUDE_CENTER)

        # Define the projection of the map.  Transverse mercator
        self.PROJ4 = f" +proj=tmerc +lat_0={self.LATITUDE_CENTER:.7f} +lon_0={self.LONGITUDE_CENTER:.7f}"
        self.PROJ4 += " +k_0=0.9996 +datum=WGS84 +units=m +no_defs "
        self.MAP = pyproj.Proj(self.PROJ4)

        self.MBTILES_PATH  = os.path.join(self.DATA_DIRECTORY, "MBTILES/MBTILES_06.mbtiles")        
        self.BASE_MAP_PATH = os.path.join(self.DATA_DIRECTORY, 'maps/seattle_base_map.tif')

        self.TIDES_PATH = os.path.join(self.DATA_DIRECTORY, 'Tides/seattle_tides2.pd')
        self.TIDES_DF = pd.read_pickle(self.TIDES_PATH)

        self.TIMEZONE = dateutil.tz.gettz('US/Pacific')

        self.STYC_MARKS = dict(
            n = lla.from_degrees_minutes((47, 41.064), (-122, 24.679)),
            b = lla.from_degrees_minutes((47, 40.285), (-122, 25.342)),
            r = lla.from_degrees_minutes((47, 44.387), (-122, 22.944)),
            u = lla.from_degrees_minutes((47, 45.676), (-122, 23.833)),
            m = lla.from_degrees_minutes((47, 41.783), (-122, 24.538)),
            w = lla.from_degrees_minutes((47, 39.617), (-122, 26.467)),
            k = lla.from_degrees_minutes((47, 35.700), (-122, 28.800)),
            d = lla.from_degrees_minutes((47, 35.933), (-122, 23.267)),
            j = lla.from_degrees_minutes((47, 44.755), (-122, 28.38)),
            o = lla.from_degrees_minutes((47, 42.600), (-122, 30.450))
        )
        self.CYC_MARKS = dict(
            n = lla.from_degrees_minutes((47, 41.064), (-122, 24.679)),
            b = lla.from_degrees_minutes((47, 40.285), (-122, 25.342)),
            r = lla.from_degrees_minutes((47, 44.387), (-122, 22.944)),
            u = lla.from_degrees_minutes((47, 45.676), (-122, 23.833)),
            m = lla.from_degrees_minutes((47, 41.783), (-122, 24.538)),
            w = lla.from_degrees_minutes((47, 39.617), (-122, 26.467)),
            k = lla.from_degrees_minutes((47, 35.700), (-122, 28.800)),
            d = lla.from_degrees_minutes((47, 35.933), (-122, 23.267))
        )
        self.RACE_MARKS = self.STYC_MARKS

    def init_styc(self):
        "Initialize for STYC races."
        self.RACE_MARKS = self.STYC_MARKS

    def init_cyc(self):
        "Initialize for CYC races."
        self.RACE_MARKS = self.CYC_MARKS

    def mark_position(self, mark):
        return self.RACE_MARKS[mark]
        
    def init_san_diego(self, logging_level=logging.INFO):
        "Localize to SanDiego."        

        self.init_j105()
        self.set_logging_level(logging_level)

        self.LOCALE = "San Diego"
        # By centering everything here, we can easily compare North/East (NED) coordinates across
        # runs and races, etc.
        self.LATITUDE_CENTER    = 32.7036
        self.LONGITUDE_CENTER   = -117.1833

        # Define the projection of the map.  Transverse mercator
        self.PROJ4 = f" +proj=tmerc +lat_0={self.LATITUDE_CENTER:.7f} +lon_0={self.LONGITUDE_CENTER:.7f}"
        self.PROJ4 += " +k_0=0.9996 +datum=WGS84 +units=m +no_defs "
        self.MAP = pyproj.Proj(self.PROJ4)

        self.MBTILES_PATH  = os.path.join(self.DATA_DIRECTORY, "MBTILES/MBTILES_10.mbtiles")        
        self.BASE_MAP_PATH = os.path.join(self.DATA_DIRECTORY, 'maps/sandiego_base_map.tif')

        self.TIMEZONE = dateutil.tz.gettz('US/Pacific')        


# All downstream code should access globals through global_variables.  
G = RaceAnalysis()
