
# These are libraries written for RaceAnalysis
from global_variables import G
import race_logs

import bokeh.io as bio
import bokeh.layouts as bla
import bokeh.plotting as bpl
import bokeh.models as bmo
import bokeh.embed as bem
import bokeh.events
import bokeh
import json


# Initialize for Seattle.
G.init_seattle(logging_level="INFO")

date = '2021-02-20'
df, race = race_logs.read_date(date, race_trim=True)

sdf = df[::200]
# sdf = df
index = sdf.index
index = sdf.row_times.dt.tz_localize(None)

colors = bokeh.palettes.Set1[9]

from bokeh.models.callbacks import CustomJS

callback = CustomJS(code="""
// the event that triggered the callback is cb_obj:
// The event type determines the relevant attributes
console.log('Tap event occurred at x-position: ' + cb_obj.x)
""")

callbackXRange = CustomJS(code="""
// the event that triggered the callback is cb_obj:
// The event type determines the relevant attributes
console.log('Xrange change occurred at: ' + cb_obj.x)
""")


ht = bmo.HoverTool(
    tooltips = [("", "@x{%H:%M:%S}, @y{%.1f}")],

    formatters={
        '@x'        : 'datetime',  # use 'datetime' formatter for '@date' field
        '@y'        : 'printf', 
    }
)

args = dict(sizing_mode="stretch_width", plot_height=350, x_axis_type='datetime')
args = dict(sizing_mode="stretch_width", plot_height=350, output_backend="webgl", x_axis_type='datetime')

def create_lines(index, df, cols, args):
    fig = bpl.figure(title=None, **args)
    for i, col in enumerate(cols.split()):
        fig.line(index, df[col], color=colors[i], legend_label=col.upper())
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.add_tools(ht)
    return fig

fig1 = create_lines(index, sdf, "spd sog", args)

fig1.js_on_event('tap', callback)
fig1.js_on_event('pan', callback)
fig1.x_range.js_on_change('end', callbackXRange)

figs = [
    fig1,
    create_lines(index, sdf, "twa awa", dict(x_range=fig1.x_range, **args)),
    create_lines(index, sdf, "tws aws", dict(x_range=fig1.x_range, **args)),
    create_lines(index, sdf, "hdg cog", dict(x_range=fig1.x_range, **args))
    ]


graphs = bla.column(*figs, width_policy="max")

if True:
    bio.output_file("panning.html")
    bio.show(graphs)
else:
    with open('/Users/viola/tmp/graphs.json', 'w') as fs:
        json.dump(bem.json_item(graphs), fs, indent=4)
