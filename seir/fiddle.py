"""Serve an interactive visualization of the model

This file uses holoviews and panel to serve a dashboard where a user can play with the simulation parameters and see how
the model behaves.
"""

from collections import OrderedDict

import holoviews as hv
import holoviews.plotting.bokeh
import pandas as pd
import panel as pn

import data
import seir

pn.extension()

slider_type_map = {
    float: (pn.widgets.FloatSlider, '0.00[0000]'),
    int: (pn.widgets.IntSlider, None),
    pd.Timestamp: (pn.widgets.DateSlider, None),
}

SLIDERS = OrderedDict()
for name, param in seir.PARAMS.items():
    slider_type, fmt = slider_type_map[type(param.default)]
    if fmt:
        SLIDERS[name] = slider_type(name=param.description, start=param.min, end=param.max, value=param.default,
                                    step=(param.max - param.min)/101, format=fmt)
    else:
        SLIDERS[name] = slider_type(name=param.description, start=param.min, end=param.max, value=param.default)


def disable_logo(plot, element):
    """Remove Bokeh logo from plots"""
    plot.state.toolbar.logo = None


@pn.depends(**{k: v.param.value for k, v in SLIDERS.items()})
def draw(**kwargs):

    t = pd.date_range(seir.DATE_OF_SIM_TIME_ZERO, '1 May 2020', freq='1d')
    s, e, i0, i0d, i1, i2, f, fd, r, rd = seir.run_odeint(t, **kwargs)

    stack = (
            hv.Area.stack(
                hv.Overlay(
                    [
                        hv.Area((t, y), label=label).options(
                            fill_alpha=1.0, line_width=2, color=color, line_color=None, framewise=True
                        )
                        for y, label, color in [  # Areas are stacked bottom up in this order
                            (fd, 'Infection-related fatalities, detected', 'black'),
                            (i0d, 'Infectious, detected', 'gold'),
                            (i1, 'Infectious, severe', 'orange'),
                            (i2, 'Infectious, critical', 'darkred'),
                            (rd, 'Recovered, detected', 'darkgreen'),
                            (f, 'Infection-related fatalities, undetected', 'grey'),
                            (e, 'Exposed, undetected', 'palegoldenrod'),
                            (i0, 'Infectious, undetected', 'lightsalmon'),
                            (r, 'Recovered, undetected', 'green'),
                            (s, 'Susceptible', 'white'),
                        ]
                    ]
                )
            )
            * hv.Curve((t, fd + i0d + i1 + i2 + rd), label='Confirmed Cases, simulated').options(
                line_dash='dashed'
            )
            * hv.Curve((t, fd), label='Known fatalities, simulated').options(
                line_dash='dashed'
            )
            * hv.Overlay([
                hv.Scatter(data.lombardia, 'data', y, label=label).options(marker='o', size=6)
                for y, label, color in [
                    ('totale_casi', 'Total cases', 'blue'),
                    ('deceduti', 'Deceased', 'red'),
                ]
            ])
            * hv.Overlay([
                hv.VLine(t).options(color='grey', line_dash='dashed', line_width=1) for t in
                [data.DATE_OF_LOMBARDY_LOCKDOWN, data.DATE_OF_SHUTDOWN_OF_NONESSENTIALS, pd.datetime.now()]
            ])
            ).options(
                title='SEIR Total Population Trace, Stacked',
                xlabel='days',
                ylabel='number of individuals, stacked',
                aspect=2,
                responsive=True,
                legend_position='top_left',
            ).redim.range(y=(1, 1.5 * max(data.lombardia['totale_casi'])))

    return (stack.options(yformatter='%d') + stack.options(logy=True, show_legend=False)).cols(1)


def fiddle():
    n_sliders = len(SLIDERS)
    slider_values = list(SLIDERS.values())
    gspec = pn.GridSpec(sizing_mode='stretch_both')
    gspec[0, 0] = pn.WidgetBox(*slider_values[:n_sliders//2])
    gspec[0, 1] = pn.WidgetBox(*slider_values[n_sliders//2:])
    gspec[1, :] = hv.DynamicMap(draw)
    return gspec


if __name__ == '__main__':
    hv.plotting.bokeh.ElementPlot.hooks.append(disable_logo)
    fiddle().servable('SEIR Simulation')
