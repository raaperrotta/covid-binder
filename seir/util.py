import holoviews as hv


def ternary(conditional, if_true, if_false):
    return if_true if conditional else if_false


def plot_trace(trace, varnames=None, tune=0):
    """Plot the distribution and trace for each latent variable in a pymc trace object.

    trace: the trace output from pymc.sample
    varnames: Optional specification of variables to include in the trace plot. If None, use all variables not ending with '_'
    """
    vline = hv.VLine(tune).options(color='grey', line_width=1, line_dash='dashed', aspect=3, responsive=True)
    plots = []
    for var in varnames or [var for var in trace.varnames if not var.endswith('_')]:
        x = trace.get_values(var, combine=False)
        if not isinstance(x, list):
            x = [x]
        plots.append(
            hv.Overlay([hv.Distribution(xi[tune:], [var], [f'p({var})']) for xi in x], group=var)
            .options(aspect=3, responsive=True)
        )
        plots.append(
            hv.Overlay([hv.Curve(xi, 'index', var).options(alpha=0.6) for xi in x] + [vline])
            .options(aspect=3, responsive=True)
        )
    return hv.Layout(plots).cols(2)
