#!/usr/bin/env python
# coding: utf-8

# # Spinnakers: Rules and Reaching
# 
# We own an untagged spinnaker that is smaller than the "class" spinnaker (74 m2 vs 89 m2), and was produced for the previous owner as a cruising kite.  There is a question if this kite would be better on tight reaches (less than 90 apparent), rather than flying the big kite which is listed as an A2 runner and designed to sail lower.
# 
# In windward/leeward racing the answer is almost certainly no.  But we often race long distances on fixed courses, which can include tight reaches for long legs.
# 
# We have some experience sailing this kite doublehanded (where it is easier to handle).  And less experience sailing with a full crew.  There is anecdotal evidence it is a better reacher... and certainly puts less load on the boat below 90 AWA when the wind is above 10 kts.
# 
# I have been consistently surprised that other boats fly their kites when reaching, perhaps down to 60 AWA (which can be done in 5-10 kts) and down to 90 in a wide range of conditions.  It is possible that the small kite will allow us to sail hotter and faster.
# 
# **Summary**: 
# - Is the spinnaker legal by rule 6.6.2 of the [J/105 CLASS ASSOCIATION RULES](http://j105.org/wp-content/uploads/2015/12/Class-Rules-2020-final.pdf)?
# - Is the spinnaker better in some conditions?

# ## Rule 6.6.2
# 
# ![im](Data/Images/rule_6.6.2.png)
# 
# 
# ```
# SA = [(luff length + leech length) * .25 * foot length]
#     + [(half width – .5 * foot length) * (leech length + luff length)] ÷ 3
# ```
# 
# 1. ("SA") shall not exceed 89 m2
# 1. where luff length shall not be greater than 15,100 mm nor less than 13,600 mm, 
# 1. leech length shall not be greater than 12,140 mm
# 1. half width shall not be less than .65 * foot length.
# 
# Lengths have been measured as described here: [ORC page on measurement](https://www.orc.org/index.asp?id=17) and [Youtube video on ORC spinnaker measurement](https://youtu.be/GL6UHyQHW0M?t=157).
# 
# 
# 
# <table style="width:100%">
#   <col width="30%">
#   <col width="70%">    
#     <tr>
#     <td>
#         <div align="left">
#             In the figure below the measurements are:            
#             <br><br>
#             - AMG: half width
#             <br> <br>
#             - SLE: leech length
#             <br> <br>
#             - SLU: luff length
#             <br> <br>
#             - ASF: foot length
#         </div>
#     </td>
#     <td>
#         <img src="Data/Images/kite_measurements.png"/>
#     </td>
#   </tr>
# </table>
# 
# 

# In[29]:


import math
from utils import DictClass

FEET_PER_METER = 3.28084
INCH_PER_METER = 39.37008

def convert_to_metric(d):
    res = DictClass()
    for (k, v) in d.items():
        feet, inch = v
        res[k] = feet / FEET_PER_METER + inch / INCH_PER_METER
    return res


# In[30]:


# Measured lengths.  (feet, inches)
kite = DictClass(luff=(48, 1), leech=(38, 9), foot=(22, 8), half=(22, 2))

# Convert to metric (since everything on the boat is metric)
mkite = convert_to_metric(kite)

print(mkite)


# In[33]:


def SA(kite):
    a = ((kite.luff + kite.leech) * 0.25 * kite.foot)
    b = ((kite.half - 0.5 * kite.foot) * (kite.leech + kite.luff)) / 3
    return a + b

print(mkite)
print()

def legal_kite(mkite):
    # ("SA") shall not exceed 89 m2
    rule1 = SA(mkite) <= 89
    print(f"Rule1:  {rule1}.  SA is less than 89: {SA(mkite):.2f}")

    # luff length shall not be greater than 15,100 mm nor less than 13,600 mm, 
    rule2a = mkite.luff <= 15.1
    rule2b = 13.6 <= mkite.luff
    print(f"Rule2a: {rule2a}.  Luff {mkite.luff:.2f} shall not be greater than 15.10")
    print(f"Rule2b: {rule2b}.  Luff {mkite.luff:.2f} shall not be less than 13.60")

    # leech length shall not be greater than 12,140 mm
    rule3 = mkite.leech <= 12.14
    print(f"Rule3:  {rule3}.  Leech {mkite.leech:.2f} shall not be greater than 12.14")

    # half width shall not be less than .65 * foot length
    rule4 = 0.65 * mkite.foot <= mkite.half
    print(f"Rule4:  {rule4}.  Half width {mkite.half:.2f} shall not be less than .65 * foot length {0.65 * mkite.foot:.2f}")

    if rule1 and rule2a and rule2b and rule3 and rule4:
        print("The kite is LEGAL.")
    else:
        print("The kite is NOT LEGAL.")

legal_kite(mkite)


# ## The kite is legal!
# 
# We don't yet know if it is a reaching design, rather than a small runner.
# 
# A true reacher has a narrow half width and a shorter luff, relative to a runner. Note that the rules require a minimum luff length and a minimum half width (which ensures that the kite has lots of volume and will not sail upwind well).

# In[42]:


# Here is a hypothetical kite, with a max luff and leech, and a wider foot and half.
hkite = DictClass(luff=15.1, leech=12.14, foot=7.91, half=7.76)
print(mkite)
print(hkite)

legal_kite(hkite)


# ## Is this smaller Kite better?
# 
# The boring answer is no, because there is less sail.  But there may be a niche for reaching, particularly on long races like foul weather bluff.
# 
# I have a set of VPP polars for the J105, computed for ORC (shown below). These polars were computed in 2011, and the rule for the 89m kite was in place for 10 years.  They show that the big kite can be used down to 45 AWA (up to 10 kts),  though in my experience the boat quickly struggles in a puff.
# 
# The real payoff will be if we can get to 45 awa in 12 or 14 knots.  
# 
# <table style="width:100%">
#   <col width="20%">
#   <col width="80%">    
#     <tr>
#     <td>
#         <div align="left">
#             There is another python notebook on Polars, but it is worth staring at this for a second (particularly 
#             when trying to sail upwind on a spinnaker).
#             <br> <br>
#             On the "true wind" side you can hold a kite down to 75 TWA, but this is not a particularly good spot.
#             You are better off either sailing higher on the jib, or lower on the kite.
#             <br> <br>
#             In practice you should not really sail much lower than 90 TWA. Though you may need to hold it though a 
#             puff or two.
#             <br> <br>
#             On the apparent wind side, the angles are much lower, down to 45 AWA.  Even though the geometry/trig of 
#             this is straightforward,  I found it a bit surprising.  I don't think about flying a kite much lower 
#             than 80 AWA.
#             <br> <br>
#             So, we should be willing to consider sailing down to 45-60 AWA, but it may not be very effective.  
#         </div>
#     </td>
#     <td>
#         <img src="Data/Images/polars_pic.png"/>
#     </td>
#   </tr>
# </table>
# 
# 
# 
# <table style="width:100%">
#   <col width="30%">
#   <col width="70%">    
#     <tr>
#     <td>
#         <div align="left">
#             One of the tricky implications of all this is what happens if the winds change.  If you are on a 90 degree 
#             TWA reach, and then get a header, it can be hard to drop the kite.  Even if you are willing to sail down, 
#             you'll eventually need to sail away from the mark.
#             <br><br>
#             No matter what, the drop at the mark could be hard.  
#             <br><br>
#             The correct drop is a leeward (a high risk maneuver).  Pull out the jib on the leeward side, and then blow the halyard.  (See below.)
#             <br><br>
#             In light winds you can do a tack-mexican, or taxican.
#             Pull out the jib on the windward side, and then tack up through wind (under kite if at all possible).  
#             Then drop, much like a mexican.
#         </div>
#       </td>
#     <td>
#         <img src="Data/Images/reach_on_kite.png"/>
#       </td>
#   </tr>
# </table>

# ### Leeward Drop
# 
# From Dave Ullman:  [LINK](http://j105.org/wp-content/uploads/2016/01/Dousing.pdf)
# 
# > The general rule we use now is: unless we absolutely have to, we
# > won’t do a leeward drop in more than 7 knots of wind. Only if
# > you’re laying the mark, or worse, if you’re overlayed, should you
# > attempt a leeward takedown.  Here’s how to do it properly. If you
# > have a very competent crew, and enough people, keep heading
# > toward the mark and blow the halyard, **release it completely**. If
# > you leave the tack nailed and the foot stretched tight, you
# > usually won’t shrimp. Like a symmetric spinnaker, the a-sail
# > should float just over the water. Then grab the middle of the
# > foot and haul the sail into the boat through the forward hatch.
# >
# > If you don’t have 100% percent confidence in your crew work,
# > you’ll want to run off for two or three boatlengths as you douse
# > the sail. This will blanket the chute behind the main,
# > depressurizing the sail so you can gather it in, under
# > control. But running off will take you away from the mark
# 
# 
# ### Letterbox?
# 
# Another option is a letterbox drop [VIDEO](https://www.uksailmakers.com/news/2019/10/30/letterbox-takedown-video-updated).  In my experience you can do a letterbox at any apparent wind angle, even with the jib furled.  
# 
# 
# ## A reach to a reach
# 
# We ran into a related issue in a recent race.  We were reaching on spinnaker at 90 TWA to the mark, tacking, and then sailing almost directly back.  We did not realize it at the time, but **we would be on spinnaker for both legs**.
# 
# This is a especially tricky situation, and almost impossible in winds above 10kts.  For 10+ I feel we should probably douse early (or never raise) and then tack through.

# In[ ]:




