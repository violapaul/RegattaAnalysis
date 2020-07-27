
#### Cell #0 Type: markdown ####################################################

#: # Sail Simulator
#: 


#### Cell #1 Type: module ######################################################

import process as p
import numpy as np

from nbutils import display_markdown, display

import nbutils

import matplotlib.pyplot as plt

#### Cell #2 Type: module ######################################################

# Vectors are numpy arrays.  We'll provide some helpers for constructing and maniuplating.

def vec(north, east):
    return np.array([north, east])

def vec_from(heading, speed):
    "By convention winds/currents that comes from the north are vectors pointing south."
    return np.array([-speed * p.north_d(heading), -speed * p.east_d(heading)])

def vec_to(heading, speed):
    "By convention winds/currents that comes from the north are vectors pointing south."
    return np.array([speed * p.north_d(heading), speed * p.east_d(heading)])

def north(v):
    return v[0]
    
def east(v):
    return v[1]
    
def length(v):
    return np.linalg.norm(v)
    
def heading_from(v):
    return np.degrees(np.arctan2(-east(v), -north(v)))

def heading_to(v):
    return np.degrees(np.arctan2(east(v), north(v)))

def polar_from(v):
    return heading_from(v), length(v)

def polar_to(v):
    return heading_to(v), length(v)

def to_vec_string(v):
    return f"<Vector: {length(v)} at {p.compass_angle(heading_to(v))}>"

def from_vec_string(v):
    return f"<Vector: {length(v)} at {p.compass_angle(heading_from(v))}>"



#### Cell #3 Type: module ######################################################

v1 = vec_from(0, 10)

display(vec_string(v1))

#### Cell #4 Type: module ######################################################

    
def arrow(pos, vec, color):
    return plt.arrow(east(pos), north(pos), east(vec), north(vec), 
                     head_width=0.5, head_length=1.0, fc=color, ec=color, length_includes_head=True)

def fig(d):
    plt.figure()
    plt.axis([-d, d, -d, d])

    

#### Cell #5 Type: module ######################################################

# Land wind
lwa = 0  # From the North or 0 degrees
lws = 10
land_wind = vec_from(lwa, lws)

# Current
curangle = 270  # From the East or 90 degrees
curspeed = 3
current = vec_to(curangle, curspeed)

# True Wind (which is sometimes/rarely called water wind).  This is the wind one would
# perceive if floating along with the current.  So if you are flowing 3 m/s to the east,
# then the wind is coming from the north west.
true_wind = land_wind - current

# Visualize the vectors.      
pos = vec(5, 0)
fig(10)
aa = []
aa.append(arrow(pos, land_wind, 'r'))
aa.append(arrow(pos, current, 'b'))
aa.append(arrow(pos, true_wind, 'g'))
npos = pos + land_wind - current
arrow(npos, current, 'b')
plt.legend(aa, "land_wind current true_wind".split())
plt.title("Wind from the North. Current from the East.")

display_markdown(f"Land Wind is {from_vec_string(land_wind)}.")
display_markdown(f"Current is {from_vec_string(current)}.")

display_markdown(f"True Wind is {from_vec_string(true_wind)}.")

#### Cell #6 Type: module ######################################################


#### Cell #7 Type: module ######################################################

twd, tws = polar_from(tw_v)

btv = vector_to(twd+45, 6)

# Visualize the vectors.      
pos = np.array([0, 0])
plt.figure()
d = 15
plt.axis([-d, d, -d, d])
aa = []
aa.append(arrow(pos, lw_v, 'r'))
aa.append(arrow(pos, cur_v, 'b'))
aa.append(arrow(pos, tw_v, 'g'))
npos = pos + lw_v - cur_v
arrow(npos, cur_v, 'b')


aa.append(arrow(npos, btv, 'cyan'))

# copy of the true wind vector
arrow(npos + btv - tw_v, tw_v, 'g')

awv = tw_v - btv  # apparent wind vector

aa.append(arrow(npos + btv - tw_v, awv, 'orange'))

nnpos = npos+btv
# Copy of apparent wind vector
aa.append(arrow(nnpos - awv, awv, 'orange'))

plt.legend(aa, "land_wind current true_wind btv apparent_wind".split(), loc='upper left')
plt.title("Wind from the North. Current from the East.")


#### Cell #8 Type: module ######################################################

# From these vectors we can compute the AWA/AWS/TWA etc.

# AWA is the angle between 


#### Cell #9 Type: module ######################################################



#### Cell #10 Type: metadata ###################################################

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
#:     "timestamp": "2020-06-14T19:35:07.677599-07:00"
#:   },
#:   "nbformat": 4,
#:   "nbformat_minor": 2
#: }

#### Cell #11 Type: finish #####################################################

