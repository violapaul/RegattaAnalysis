
# Wind scales

Beaufort:

- 0: 0kts 
- 1: 1-3kts
- 2: 4-6
- 3: 7-10
- 4: 11-16
- 5: 17-21
- 6: 22-27

# Need a better code overview model

Figure out how to hide everything but python defs and classes.

hideshow mode is good, but it leave comments.

See also https://www.gnu.org/software/emacs/manual/html_node/elisp/Overlays.html#Overlays

# Build a simple sailing simiulator!

- Polars for speed
- Leeway 
- Current 

Explore how this effects TWA/TWD, and other algorithms.

And explore how it works for low winds (where my algorithms don't currently work well!).

# Figure out VMG versus VMC

What is the best angle to sail to a mark that is not upwind?

Assume wind is coming from 0, and mark is at 0.  Clearly if you are sailing directly
upwind, then you sail 45 degree off.

But if you are sailing to a mark at 10, do you sail at 50, or 45.  How about mark at
50?

In the past I thought we established that if you can fetch the mark, then sail
directly toward it.  But if you can't then sail VMG, and tack.

Or, **when is the shortest path not the shortest time**?

And you can ask the same for downwind marks.

https://en.wikipedia.org/wiki/Velocity_made_good

Note, the basic math here should be pretty straightforward.



# True Wind, over ground or over water?

Current is the real difference between SOG and SPD.  With 1-2 kts of current it can
make a big difference.

I get that you would want to know TWS wrt to the water.  If you are seeing,  but do you what about TWD?

If you use SPD/HDG/AWA to compute TWD, then it will change on the two tacks?  Right?  Or not.

Need to write a simulator of some kind...  it would be good to test the inference code in any case.

## Why can’t I use SOG and COG for calculating True Wind? 

From `Sailboat/Articles/essential_guide_instrument_BnG.pdf`

If you substitute SOG (Speed Over Ground) and COG for Boat speed and Heading in the
calculation of True Wind Direction, you are negating the fact that when sailing on
waters with tides or currents you are, in effect, travelling on a ‘moving carpet’.

Clearly if you were to take the sails down and stop the boat you may still be moving
over the ground because of the tide but you would be stationary relative to the
water! This is where the distinction between True Wind and Ground Wind comes into
play.

The wind speed and direction when measured relative to a fixed point on the ground is
given the term Ground Wind, while the wind speed and direction measured relative to
the water is given the term True Wind.  When sailing, you are more interested in the
affect the wind has on the boat rather than measurements from a purely meteorological
point of view, so True Wind is the preferred choice. Racing navigators constantly
convert between True Wind and ground wind to determine the accuracy of weather
forecasts which use Ground Wind.

If you have a large difference between the speed of the boat and the speed of the
tide (tide rate) then it is possible to use SOG without many issues (e.g. a
maximultihull travelling at 40kt in 1kt tide is not hugely interested in the tide
rate effects), however if you are sailing a more normal boat (say 8kt in 1kt tide)
then you should stick with boat speed (speed through the water) so that you have a
clear understanding of the effect of tide on your boat.

The units of Wind Speed in instrument systems are usually measured in knots (usually
abbreviated kt or kn). However sometimes it is useful to display wind speeds in
Beaufort Scale numbers – this allows “at a glance” checking, sometimes preferred by
those who cruise and aren’t too interested in the difference between 18 and 19 knots.

# Literate Notebooks

## Can we merge changes made in the literate module back into the notebook.

- We have the cell numbers, try not to screw with them.
  - Just pull them back in.
  - Take the diff to make sure the changes are not too massive (wrong cells, etc).

## Can we directly import literate notebooks?

Perhaps there is something in the `importlib` module that would help.

# Create a query language for sailing "events"

Needed to find tacks, jibes, long close haul runs, mark roundings, etc.

What is a race tack? 

- Minimum of 30 seconds close hauled on one tack.
- Round up and come on to the other tack.
- At least 30 seconds on second tack.

Can this be done simply and efficiently?

# YAPF for code re-formatting

    https://github.com/google/yapf

Made some changes to yapf.ini in sailing.

# Add boat compare when other boat is available

- Perhaps from AIS?
- What do we compare?
  - VMG, 

# Match the images we've taken on the boat with the rest of the data!

Exiftool can extract the date/time and GPS position!!

# Reorganize code so that it loads better

https://docs.python.org/3/reference/import.html#package-relative-imports

Use . and .. with imports!

# Figure out naming of "literate notebooks".

Capitalized vs. downcase has some issues on Mac, since my current drive is case
insenstive (APFS).  Foo, foo, and FOO are all the same file!

Extensions help, but they are not everything.

# Makefiles for LN?

# Process videos automatically

  - Estimate pose versus time.  AHRS or complementary filter?

Let's start with a complementary filter.

https://www.pieter-jan.com/node/11
https://www.pieter-jan.com/node/7

The acceleromter removes the drift from the down vector.  And the magnetomter removes
the drift in the north vector.


# Better Graphing

- Align the graphs with a map, including clipping
- 

# Get the matplotlib GUI to work in Jupyter...  where are the events going?

After a while it just dies.

# Race Analysis TODO

So much to do, so little time!

## Round trip notebooks in Markdown?

- Would be much nicer for GIT.
  - Diffs
  - Files are smaller
  - Easier to search all code for references (to refactor).


## How to keep notebooks and libraries in sync?

- Often there is a notebook for each major library.
- The notebook includes code and clear documentation for that code.
- But the library has the same (or similar) code.
  - It can get out of sync.
  - Keeping them updated in sync is hard!

Does this have something to do with Markdown roundtrip?

**Can I do literate programming?**

I.E. code and doc which are written together.  This would argue that the notebook is
the library.  What would that mean?

The notebook has graphs, links to other resources, it loads and processes data.

The notebook **runs**.

The library is more streamlined.  It just has functions, perhpas constants, etc.

Should the library be extracted from the notebook?  What would be extracted?

- A subset of the code includes
- A subset of the declarations
- A subset of the functions/classes
- No inline code.

Which is the primary source?



## Produce race videos with overlays!

Issues:

- Lots of data!!  36GB an hour!
- Takes a long time.  Latest mac encoder is pretty good (a bit faster than realtime).
  - Could make this part of pulling the data off the SD card.
- Need to produce the overlays, graphs, numbers, etc.
- Any particular set of overlays will be limited.  We won't have the option to easily
  change what is shown.
- Stabilization and horizon correction.  Virb Edit does this, but its painful to use.
  I want to automate this step!

- Just how much can we upload to Youtube??  Currently the videos are 36GB an hour!

### Alternatives.

- A coupled player where I could keep the data locally (video and telemtry) and
  simply display the data simultaneously.
  - The graphs could easily change.
  - No upload of data.
  - Can't be easily shared.

Or is it just better to include a viewer in the UX.

Damn, this is hard.




