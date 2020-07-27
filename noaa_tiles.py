# Load some libraries
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import numpy as np
import math
import itertools as it
import logging

import requests
from PIL import Image
from io import BytesIO

import concurrent.futures

# These are libraries written for RaceAnalysis
import global_variables
# logging.getLogger().setLevel(logging.DEBUG)
import race_logs
import process as p

G = global_variables.init_seattle(logging_level=logging.DEBUG)
TILE_SIZE = 256

def fetch_noaa_chart(df, zoom=14, border=0.2):
    # Add/sub just a bit to make the map interpretable for small excursions
    lat_max, lat_min, lat_mid = p.max_min_mid(df.latitude, border) + (0.002, -0.002, 0.0)
    lon_max, lon_min, lon_mid = p.max_min_mid(df.longitude, border) + (0.002, -0.002, 0.0)
    
    x_min, y_max = deg2num(lat_min, lon_min, zoom)  # South West corner
    x_max, y_min = deg2num(lat_max, lon_max, zoom)  # North East corner

    addresses = list(it.product(range(x_min, x_max), range(y_min, y_max), [zoom]))

    logging.info(f"About to fetch {len(addresses)} tiles.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        addressed_tiles = list(ex.map(fetch_noaa_tile, addresses))
    
    cols = x_max - x_min
    rows = y_max - y_min
    res = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE, 3), dtype=np.uint8)

    for (x, y, zoom), tile in addressed_tiles:
        coffset = (x - x_min) * TILE_SIZE
        roffset = (y - y_min) * TILE_SIZE
        logging.debug(f"{roffset}, {coffset}")
        res[roffset:(roffset+TILE_SIZE), coffset:(coffset+TILE_SIZE), :] = tile

    return res

def fetch_noaa_tile(address):
    logging.info(f"Fetching tile: {address}")
    x, y, zoom = address
    r = requests.get(noaa_url(x, y, zoom))
    tile = Image.open(BytesIO(r.content))
    if tile.mode != 'RGB':
        np_tile = np.asarray(tile.convert(mode="RGB"))
    else:
        np_tile = np.asarray(tile)
    logging.info(f"Done fetching tile: {address}")
    return address, np_tile
    
def noaa_url(x, y, zoom):
    return f"https://tileservice.charts.noaa.gov/tiles/50000_1/{zoom}/{x}/{y}.png"    

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    # This returns the NW-corner of the square. Use the function with xtile+1 and/or
    # ytile+1 to get the other corners. With xtile+0.5 & ytile+0.5 it will return the
    # center of the tile.
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

border = 0.2
zoom = 14

dfs, races, big_df = race_logs.read_dates(["2019-11-16"])
# dfs, races, big_df = race_logs.read_dates(["2019-11-16"])
df = dfs[0]
races




import logging
from global_variables import G
G.init_seattle()
