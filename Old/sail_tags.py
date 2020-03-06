

import os

import copy
import math
import itertools as it
import importlib

import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from numba import jit

from pyproj import Proj
import cv2

import boat_shape
from boat_shape import OUTLINE as BOAT_OUTLINE

import process as p
from utils import DictClass

res = pd.read_html("/Users/pviola/Downloads/Sail Tag Registration List - J_105 Class Association.htm")
