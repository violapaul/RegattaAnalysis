# Basics
import os
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cv2

# These are libraries written for RegattaAnalysis
from global_variables import G  # global variables
from latlonalt import LatLonAlt as lla
import utils
from utils import DictClass

import race_logs                # load data from races

import nbutils
from nbutils import display_markdown, display

G.init_seattle()

TIDE_PRINT_DIR = "Data/Currents/TidePrints"
TIDE_PRINT_FILES = [f"im-{i:03d}.jpg" for i in range(24, 33)]
TIDE_PRINT_PATHS = [os.path.join(TIDE_PRINT_DIR, f) for f in TIDE_PRINT_FILES]

def capture_tide_print_landmarks():
    """
    This is a one-time function that is run to capture the landmarks on a set of maps.
    Once it is done the results should be stored away in datastructure.
    """
    return capture_landmarks(TIDE_PRINT_PATHS)


def capture_landmarks(images, block=True):
    "Capture landmarks for a sequence of images."
    res = []
    for path in images:
        res.append(pick_image_landmarks(path))
        # Since previous function is asynchronous and returns immediately, you need to
        # block before bringing up the next image.
        if block:
            plt.show(block=True)
    return res


def pick_image_landmarks(image_path):
    """
    Load image, display, and collect clicked points.

    Result: dict that contains points and file names.

    Asynchronous function, which returns immediately.  Results are updated after clicking.    
    """
    path, file = os.path.split(image_path)
    G.logger.info(f"Processing file {file}")
    im = cv2.imread(image_path)
    if im is not None:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        display("Pick the points left to right, starting at the bottom.")
        res = pick_points(im)
        # Block until window is closed.
        res.file = file
        res.path = image_path
        return res
    else:
        raise Exception(f"Missing file {path}")


def pick_points(im, fig=None):
    """
    Pick points on an image and collect the points into a list.  Coordinates are in pixels.
    Asynchronous function, which returns immediately.  Results are updated after clicking.

    Collects a numpy array, rows are points, col 0 is x, col 1 is y.    

    Returns a result, a dict such that result['points'] = collected points.

    Note, x goes from east to west.  And y goes from north to south.
    """
    display("If you want to replace a point, simply click nearby the existing point.")
    fig, ax = plt.subplots(1, 1, num=None)
    fig.tight_layout()
    ax.imshow(im)
    drawn_points = ax.plot(0, 0, linestyle = 'None', marker='+', markersize=100, color='red')[0]
    # Python's handling of lexical scoping is strange.  The callbacks below need to modify
    # the points, by wrapping it in a dict, we are not reassigning a lexical variable.
    result = DictClass(points=None)
    result.height, result.width, _ = im.shape

    def add_point(x, y, min_distance=200):
        "Add a point, but if its closer that MIN_DISTANCE, then replace existing."
        new_row = np.array([[x, y]])
        current_points = result.points
        display(current_points)
        if current_points is None:
            result.points = new_row
        else:
            distances = np.sqrt(np.square(current_points - new_row).sum(1))
            small_distances = distances < min_distance
            if np.any(small_distances):
                close_row = np.argmax(small_distances)
                G.logger.info(f"Replacing row {close_row}")
                current_points[close_row, :] = new_row
                print(current_points)
            else:
                G.logger.info(f"Adding row {new_row}")
                result.points = np.append(current_points, new_row, axis=0)

    def onpress(event):
        # Only pick the points if shift clicking
        if event.key == 'shift':
            G.logger.info(f"you pressed, {event.button}, {event.key}, {event.xdata}, {event.ydata}")
            add_point(event.xdata, event.ydata)
            x = result.points[:, 0]
            y = result.points[:, 1]
            drawn_points.set_data(x, y)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onpress)
    return result

# Based on examination of the base map, these are the lat/lon coordinates of visible
# crossing.  In order from left to right, bottom first (or equivalently first southernmost
# first, and then from west to east).

LANDMARKS = [lla.from_degrees_minutes(lat_dm=(47, 20), lon_dm=(-122, 30)),
             lla.from_degrees_minutes(lat_dm=(47, 20), lon_dm=(-122, 20)),
             lla.from_degrees_minutes(lat_dm=(47, 30), lon_dm=(-122, 40)),
             lla.from_degrees_minutes(lat_dm=(47, 30), lon_dm=(-122, 20)),
             lla.from_degrees_minutes(lat_dm=(47, 40), lon_dm=(-122, 40)),
             lla.from_degrees_minutes(lat_dm=(47, 40), lon_dm=(-122, 20)),                  
             lla.from_degrees_minutes(lat_dm=(47, 50), lon_dm=(-122, 20))]

TIDE_PRINT_DIR = "Data/Currents/TidePrints"
TIDE_PRINT_FILES = [f"im-{i:03d}.jpg" for i in range(24, 33)]
TIDE_PRINT_PATHS = [os.path.join(TIDE_PRINT_DIR, f) for f in TIDE_PRINT_FILES]

# Set of tide print images, and points picked at the intersection of visible lat/lon
# lines.  The points are in <x, y> or <east, south> (the upper right corner is x=0, y=0,
# so increasing x moves east, and increasing y moves south).  The point order is lowest
# first, and then left to right (or southernmost first, and then west to east).
LANDMARKED_IMAGES = [
    {'points': [[866.8971738496723, 2187.633598589958],
                [1270.8041320634777, 2187.633598589958],
                [467.12376637782114, 1598.8978714859172],
                [1270.213624814627, 1599.488378734768],
                [465.3522446312693, 997.1709849071635],
                [1268.442103068075, 997.7614921560141],
                [1266.6705813215235, 400.758663568065]],
     'file': 'im-024.jpg',
     'path': 'Data/Currents/TidePrints/im-024.jpg',
     'height': 2210,
     'width': 1687},
    {'points': [[632.4514473323952, 2183.818054285506],
                [1029.419738836522, 2185.001268269125],
                [236.0747628200781, 1592.211062476078],
                [1030.6029528201407, 1596.9439184105536],
                [236.66636981188753, 991.7299657895089],
                [1031.786166803759, 996.4628217239843],
                [1032.3777737955684, 399.5313669882713]],
     'file': 'im-025.jpg',
     'path': 'Data/Currents/TidePrints/im-025.jpg',
     'height': 2208,
     'width': 1680},
    {'points': [[862.1767610847924, 2173.004836878175],
                [1265.287930861859, 2173.717047425484],
                [463.3388545915819, 1584.0067142533799],
                [1266.0001414091682, 1586.143345895308],
                [461.9144334969631, 983.6132228716363],
                [1265.287930861859, 984.3254334189455],
                [1262.4390886726217, 386.78078422643887]],
     'file': 'im-026.jpg',
     'path': 'Data/Currents/TidePrints/im-026.jpg',
     'height': 2207,
     'width': 1684},
    {'points': [[638.520293439536, 2179.912388891763],
                [1036.660839627184, 2180.5423581104146],
                [241.63968568919074, 1589.6312310154563],
                [1035.4009011898813, 1592.781077108713],
                [242.89962412649345, 986.7506887661223],
                [1036.0308704085328, 992.4204117339843],
                [1038.5507472831382, 396.46953088981525]],
     'file': 'im-027.jpg',
     'path': 'Data/Currents/TidePrints/im-027.jpg',
     'height': 2205,
     'width': 1685},
    {'points': [[846.2342766313084, 2188.8551278277537],
                [1249.5768789040183, 2191.424188988726],
                [449.9565925512732, 1598.6133260942815],
                [1252.145940064991, 1603.7514484162266],
                [451.24112313175914, 998.7375450071617],
                [1253.4304706454768, 1001.9488714583775],
                [1253.4304706454768, 403.9998862419875]],
     'file': 'im-028.jpg',
     'path': 'Data/Currents/TidePrints/im-028.jpg',
     'height': 2208,
     'width': 1674},
    {'points': [[643.2031173650494, 2182.5686137273547],
                [1041.211401947505, 2180.405525224189],
                [241.58968527731798, 1596.371629369499],
                [1036.1641954401189, 1592.7664818642234],
                [237.9845377720422, 994.3119959884294],
                [1033.2800774358982, 993.5909664873743],
                [1029.6749299306223, 396.57853961369096]],
     'file': 'im-029.jpg',
     'path': 'Data/Currents/TidePrints/im-029.jpg',
     'height': 2204,
     'width': 1658},
    {'points': [[859.5546699632837, 2182.302318523807],
                [1263.747359447885, 2183.48937488059],
                [463.07784679777177, 1592.9288373810627],
                [1265.5279439830595, 1594.709421916237],
                [463.0778467977718, 991.0912644920971],
                [1265.5279439830595, 994.652433562446],
                [1265.5279439830595, 396.9695579222208]],
     'file': 'im-030.jpg',
     'path': 'Data/Currents/TidePrints/im-030.jpg',
     'height': 2203,
     'width': 1680},
    {'points': [[650.9656029235393, 2190.295434823486],
                [1047.2390997663624, 2189.0964227301642],
                [249.89605770742952, 1602.7795090958514],
                [1043.0425574397364, 1601.5804970025297],
                [245.88296317970742, 1002.8820277075587],
                [1039.734264181966, 1001.5622416709797],
                [1036.4347990905185, 405.67884615556864]],
     'file': 'im-031.jpg',
     'path': 'Data/Currents/TidePrints/im-031.jpg',
     'height': 2207,
     'width': 1665},
    {'points': [[843.9021975656472, 2175.9663282332654],
                [1247.7448116920702, 2179.191908218301],
                [449.7363233943299, 1583.7498429807792],
                [1252.905739668127, 1589.5558869538427],
                [451.94552536319543, 983.9061287309339],
                [1254.8002642882593, 989.2230475317622],
                [1254.135649438156, 391.06968243858546]],
     'file': 'im-032.jpg',
     'path': 'Data/Currents/TidePrints/im-032.jpg',
     'height': 2201,
     'width': 1680}]

def check_points(landmarked_image):
    """
    Display the map and then plot the points.  Color (from red to yellow) is used to show
    the 'order' of the points.
    """
    pick = DictClass(**landmarked_image)
    im = cv2.imread(pick.path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(8, 12), num=None)
    fig.tight_layout()
    ax.imshow(im)
    x = [p[0] for p in pick.points] 
    y = [p[1] for p in pick.points]
    colors = [matplotlib.cm.autumn(v) for v in np.linspace(0, 1, len(x))]
    ax.scatter(x, y, 300, marker='+', color=colors)

def check_points_all(landmarked_image_list):
    print("Close each image in turn, after checking.")
    for pick in landmarked_image_list:
        display(f"Displaying {pick['file']}.  Close plot to proceed to next.")
        check_points(pick)
        plt.show(block=True)

def add_homogenous(vecs):
    """
    Add a final column of all 1's, the 'homogenous coordinate'.  This linearizes
    transforms that involve shifts.
    """
    r, c = vecs.shape
    return np.hstack((vecs, np.ones((r, 1))))

def estimate_geo_transform(landmarked_image, lat_lon_coordinates):
    """
    Given a set of landmarks in an image AND a set of corresponding lat/lon coorinates,
    estimate the transformation from pixel coordinates to lat/lon.

    Note we will use homogenous coordinates, which ensures a two way transformation.
    """
    li = DictClass(**landmarked_image)
    pts = np.array(li.points)
    inputs = add_homogenous(pts)
    lat_col = [l.lat for l in LANDMARKS]
    lon_col = [l.lon for l in LANDMARKS]
    outputs = add_homogenous(np.array([lon_col, lat_col]).T)
    model, residuals, rank, singular_values = np.linalg.lstsq(inputs, outputs, rcond=None)
    G.logger.info(f"Found a model residuals {residuals}")
    return model

# The more I look the more I discover about GDAL (got to figure out the Python bindings).
# Most recently I was looking to "geo register" the tide prints using gdal_transform,
# which in fact can be done like this:
#
# "gdal_translate -of GTiff -a_ullr {ullon} {ullat} {lrlon} {lrlat} -a_srs EPSG:4269 input.jpg output.tif"
#
# You specify the lat/lon of the upper left and lower right and the projection of the
# input file.  To be honest EPSG:4269 is a guess.

# This command does two things, first it adds the geo info to the TIFF header.  Second is
# transforms the file to GeoTiff.  Why GeoTiff?  The standard allows you to embed this
# info inside the file.  You can also do the "same" think to a JPG file.
#
# Note, the 
#
# "gdal_translate -of JPEG -a_ullr {ullon} {ullat} {lrlon} {lrlat} -a_srs EPSG:4269 input.jpg output.tif"

# The GTiff formal is GeoTiff, which can embed 

def georeference_image(landmarked_image, landmarks):
    li = DictClass(**landmarked_image)
    G.logger.info(f"Procssing {li.file}")
    model = estimate_geo_transform(li, landmarks)
    corners = np.array(
        [[0, 0, 1],          # left/upper,  West/North
         [0, li.height, 1],  # left/lower,  West/South
         [li.width, 0, 1],   # right/upper, East/North
         [li.width, li.height, 1]  # right/lower, East/South
        ]
    )
    corners_lonlat = np.dot(corners, model)
    ullon = corners_lonlat[0, 0]
    ullat = corners_lonlat[0, 1]
    lrlon = corners_lonlat[3, 0]
    lrlat = corners_lonlat[3, 1]
    directory, filename  = os.path.split(li.path)
    base, ext = os.path.splitext(li.file)
    output_file = f"geo_{base}.tiff"
    command = f"gdal_translate -of GTiff"
    command += f" -a_ullr {ullon} {ullat} {lrlon} {lrlat} -a_srs EPSG:4269"
    command += f" {li.path} {os.path.join(directory, output_file)}"
    utils.run_system_command(command)
    return command

if False:
    for li in LANDMARKED_IMAGES:
        georeference_image(li, LANDMARKS)
    
def example_race(date="2019-11-16"):
    dfs, races, big_df = race_logs.read_dates([date], race_trim=False)
    display(races)
    return dfs[0]

df = example_race("2020-05-09")
df = example_race("2020-05-18")
df = example_race("2020-04-28")

import chart
from chart import extract_region, gdal_extract_chart


TP_DIR = "/Users/viola/Python/sailing/Data/Currents/TidePrints"

sara = list(range(0, 9))

for file_num in [30, 31, 32]:
    n = file_num
    tp_file = f"geo_im-{n:03d}.tiff"
    TP_PATH = os.path.join(TP_DIR, tp_file)
    border = 1.0
    pixels = 1000
    region = extract_region(df, border)
    ch = region.union(dict(proj=G.PROJ4, border=border, pixels=pixels))
    ch = chart.gdal_extract_chart(ch, TP_PATH, "/tmp/mbtile.tif")
    image = cv2.imread(ch.path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ch = ch.union(dict(image=image))

    ch.fig = plt.figure()
    ch.fig.clf()

    ch.ax = ch.fig.add_subplot(111)
    ch = chart.draw_chart(ch)
    ch.ax.set_title(f"{tp_file} + Sara: {n-24}")
    # ch = chart.draw_track(df, ch, color='green')
    # ch = chart.draw_track(tdf, ch, color='red')    
    ch.fig.tight_layout()
    ch.fig.savefig(f"tide_print_{n-24}.pdf", orientation='portrait')
    

chart.plot_chart(df)


t1 = arrow.get('2020-05-09 12:00:00.00-07:00').datetime
t2 = arrow.get('2020-05-09 14:00:00.00-07:00').datetime

t1 = arrow.get('2020-05-18 18:00:00.00-07:00').datetime
t2 = arrow.get('2020-05-18 19:00:00.00-07:00').datetime

tdf = df.loc[(df.row_times > t1) & (df.row_times < t2)]


import xmltodict


GDAL_SIDECAR = """
<PAMDataset>
  <SRS>GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]]</SRS>
  <GeoTransform> -1.2285634038489967e+02,  4.1828964984150203e-04,  0.0000000000000000e+00,  4.7941536757695253e+01,  0.0000000000000000e+00, -2.7862682847760843e-04</GeoTransform>
  <Metadata domain="IMAGE_STRUCTURE">
    <MDI key="COMPRESSION">JPEG</MDI>
    <MDI key="INTERLEAVE">PIXEL</MDI>
    <MDI key="SOURCE_COLOR_SPACE">YCbCr</MDI>
  </Metadata>
  <Metadata>
    <MDI key="COMMENT">PFU ScanSnap Manager 01</MDI>
  </Metadata>
  <PAMRasterBand band="1">
    <Metadata domain="IMAGE_STRUCTURE">
      <MDI key="COMPRESSION">JPEG</MDI>
    </Metadata>
  </PAMRasterBand>
  <PAMRasterBand band="2">
    <Metadata domain="IMAGE_STRUCTURE">
      <MDI key="COMPRESSION">JPEG</MDI>
    </Metadata>
  </PAMRasterBand>
  <PAMRasterBand band="3">
    <Metadata domain="IMAGE_STRUCTURE">
      <MDI key="COMPRESSION">JPEG</MDI>
    </Metadata>
  </PAMRasterBand>
</PAMDataset>
"""

xxx = xmltodict.parse(GDAL_SIDECAR)
xxx['PAMDataset']['GeoTransform']
