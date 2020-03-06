"""
File contains all the global constants that are related to a specific boat, location,
or NMEA network.


Edit this file to match you boat and location.
"""

import os

from pyproj import Proj
import pandas as pd

from utils import DictClass

# Setup some global variables ################
class RaceAnalysis:
    "Global variables that are shared by most foreseeable uses of the RaceAnalysis software."
    def __init__(self):
        self.PGNS = pd.DataFrame([
            (126992, 'System Time'),
            (127245, 'Rudder'),
            (127250, 'Vessel Heading'),
            (127251, 'Rate of Turn'),
            (127257, 'Attitude (pitch, roll)'),
            (127258, 'Magnetic Variation'),
            (128259, 'Speed'),
            (128267, 'Water Depth'),
            (129025, 'Position, Rapid Update'),
            (129026, 'COG & SOG, Rapid Update'),
            (129029, 'GNSS Position Data'),
            (129539, 'GNSS DOPs'),
            (129540, 'GNSS Sats in View'),
            (130306, 'Wind Data')
        ], columns=['id', 'Description'])
        
        self.PGN_WHITELIST = self.PGNS.loc[:, 'id'].tolist()
        
        self.SAMPLES_PER_SECOND = 10
        self.MS_2_KNOTS = 1.944

        self.DATA_DIRECTORY = 'Data'
        self.LOGS_DIRECTORY = 'Data/Logs'
        self.MAP_DIRECTORY = 'Data/Maps'
        self.LOG_INFO_PATH = os.path.join(self.DATA_DIRECTORY, 'log_info.pd')


class PeerGynt(RaceAnalysis):
    "Global variables that are shared for all races sailed by Peer Gynt, J105"
    def __init__(self):
        super().__init__()

        self.DEVICES = pd.DataFrame([
            (0,   "Actisense"                            ),
            (1,   "BT-1"                                 ),
            (2,   "Triton Autopilot Keypad"              ),
            (3,   "Autopilot"                            ),
            (4,   "RC42 Rate Compass"                    ),
            (5,   "Zeus iGPS"                            ),
            (6,   "Zeus Pilot Controller"                ),
            (7,   "Zeus Navigator"                       ),
            (8,   "V50 Radio"                            ),
            (9,   "Triton Display"                       ),
            (10,  "Triton Display"                       ),
            (11,  "Triton Display"                       ),
            (12,  "Triton Display"                       ),
            (13,  "Triton Display"                       ),
            (14,  "Wind Sensor"                          ),
            (16,  "Zeus MFD"                             ),
            (35,  "Airmar Depth/Speed Transducer DST200" ),
            (43,  "NAIS-400"                             ),
            (127, "ZG100 Antenna"                        ),
            (128, "ZG100 Compass"                        )
        ], columns=['src', 'Device'])


class Seattle(PeerGynt):
    "Global variables unique to Seattle."
    def __init__(self):
        super().__init__()
        # By centering everything here, we can easily compare North/East (NED) coordinates across
        # runs and races, etc.
        self.LATITIDE_CENTER    = 47.687307
        self.LONGITUDE_CENTER = -122.438644

        # These are chosen by hand to in order to cover the entire region
        self.LAT_MAX_MIN = (48.3, 47.3)
        self.LON_MAX_MIN = (-122.3, -122.8)

        # MBTILES_FILE = os.path.join(DATA_DIRECTORY, "MBTILES/MBTILES_06.mbtiles")

        # Define the projection of the map.  Transverse mercator
        self.PROJ4 = f" +proj=tmerc +lat_0={self.LATITIDE_CENTER:.7f} +lon_0={self.LONGITUDE_CENTER:.7f}"
        self.PROJ4 += " +k_0=0.9996 +datum=WGS84 +units=m +no_defs "
        self.MAP = Proj(self.PROJ4)

        self.BASE_MAP_PATH = os.path.join(self.MAP_DIRECTORY, 'seattle_base_map.tif')


class SanDiego(PeerGynt):
    "Global variables unique to SanDiego."
    def __init__(self):
        super().__init__()
        # By centering everything here, we can easily compare North/East (NED) coordinates across
        # runs and races, etc.
        self.LATITIDE_CENTER    = 32.7036
        self.LONGITUDE_CENTER   = -117.1833

        # These are chosen by hand to in order to cover the entire region
        self.LAT_MAX_MIN = (32.7554, 32.6176)
        self.LON_MAX_MIN = (-117.105, -117.2616)

        # self.MBTILES_FILE = os.path.join(DATA_DIRECTORY, "MBTILES/MBTILES_10.mbtiles"),

        # Define the projection of the map.  Transverse mercator
        self.PROJ4 = f" +proj=tmerc +lat_0={self.LATITIDE_CENTER:.7f} +lon_0={self.LONGITUDE_CENTER:.7f}"
        self.PROJ4 += " +k_0=0.9996 +datum=WGS84 +units=m +no_defs "
        self.MAP = Proj(self.PROJ4)

        self.BASE_MAP_PATH = os.path.join(self.DATA_DIRECTORY, 'sandiego_base_map.tif'),


class Uninitialized():
    def __getattr__(self, key):
        messages = ["Global variables are not yet initialized.  Please call one of the init functions first!",
                    "    For example global_variables.init_seattle() "]
        raise Exception("\n".join(messages))


# Setup a global class that can be used to access these variables. ################

# All downstream code should access globals through global_variables.G.
G = Uninitialized()

def initialzize_global_variables(variables):
    global G
    if type(G) == type(variables):
        return
    if type(G) is Uninitialized:
        G = variables
    else:
        raise Exception("Do not initialized global variables twice.  Can lead to unexpect behaviors.")

def init_seattle():
    "Rebind globals to Seattle."
    initialzize_global_variables(Seattle())
    return G

def init_san_diego():
    "Rebind globals to San Diego."
    initialzize_global_variables(SanDiego())
    return G

