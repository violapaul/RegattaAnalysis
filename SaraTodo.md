# Todo: Ideas for Using Data to Improve Racing Skills 


- [ ] Current Maps of Shilshole Bay
- [ ] Optimal Tacking
- [ ] Jibe Angles
- [ ] Jib Car Experiements
- [ ] Shroud Tension Settings 
- [ ] Relating Twist to Sail Log Data
- [ ] Compute Your Own Polars



# Current Maps 

The Puget Sound is a long network of ocean waterways bounded on the
north by the Oympic Pennisula, the Straight de Juan de Fuca, Whidbey
Island, and Deception Pass and proceeding south for 80 miles to the
southern end of Olympia, the capitol of Washington State. It is part
of the bigger Salish Sea Basin which includes the inland sea around
Vancouver Island as well. The tidal flow of water from the Puget Sound
drains in and out through Admirality Inlet and Deception Pass creating
strong currents at certain predictable times. As sailors, we need to
understand these currents well enough to take advantage of them when
we can and minimze the negative impact of adverse current when we need
to.

![Puget Sound](https://www.eopugetsound.org/sites/default/files/topical_articles/images/PugetSoundBoundaries_Basins.png)  "Image
Credit: Encyclopedia of Puget Sound"

If we sail 100 days in a year and collect data, we should be able to
find a good model of the local current at most times each day and in
most locations. There will be tricky spots known as singularities in
the current flow pattern, but we will assume we won't be stuck in a
tricky spot very long.  We want to know the flow patterns detectable
over on order of 100 meters. With additional data, we could the
improve accuracy of the model and focus in on strong current flow
areas.

**Goals**: Given the nautical charts of Shilshole Bay, plot 3
dimension current flow vectors with a magnitude and direction at each
sailed location sampling every 5 minutes.  The third dimension is
determined by time.  We don't want to use GPS time or clock time,
we want to use a measure of time with respect to the tides.  

There are two high tides and two low tides each day. One is noticibly
bigger than the other and they alternate. The textbook time interval
breaks up the tide patterns into 8 distinct points on the curve over
the course of the cycle with two highs and two lows.  We need
something a bit more precise.  In addition, we need to understand how
the high and low tide marks relate to currents. If we input the two
highs and two lows for the day into our model, we hope to get a more
accurate flow chart from our data. We might end up with more like 10
distinct day types and 10 points on the curve so 100 charts.  This
data is too big for publication in traditional methods but easy to
have on a computer or phone screen.


# Optimal Tacking

Did you tack too far over or not far enough?  Well, that depends on
the boat, the goals of the day, the crew, the current, the wind,
etc. Tacking takes time, it slows down your boat, it comes with some
risk of misexecution, and it's necessary to get from point A to point
B without hitting things like rocks and other boats. 

**Goal** If we know the wind speed and we have minimal current, how
many degrees should we turn in a tack and how fast should we take the
turn in the beginning, middle and end of the tack.  Every olympic
sailor seems to have a thoughtful answer to these questions, but it's
hard to even begin to talk about the subtly of the anatomy of a
tack. The goal here is to identify key parts of the tack that we can
measure and then control out on the water. What parts of the tack are
most important for success?  How can we measure the success of a tack?
Can we visual 10 really good tacks vs 10 tacks each of which are
flawed in some way? 






