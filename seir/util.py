import logging
import os

import multiprocessing as mp
import pickle

import holoviews as hv
import numpy as np
from tqdm import tqdm

LOG = logging.getLogger(__name__)


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


def fit_nevergrad_model(instrumentation, budget, optimizer_class, score_fn,
                        num_processes=os.cpu_count(), num_workers=os.cpu_count(), save_after=None):
    LOG.info('Computing score of default instrumentation values')
    args, kwargs = instrumentation.value
    trial_score = score_fn(*args, **kwargs)
    LOG.debug('Trial score: %s', trial_score)

    optimizer = optimizer_class(
        instrumentation, budget=budget, num_workers=num_workers,
    )
    LOG.debug('Created an optimizer of type %s with %d workers on %d processes and a budget of %d',
              optimizer_class, num_workers, num_processes, budget)
    try:
        with mp.Pool(processes=num_processes) as pool, tqdm(total=budget) as pbar:
            n_complete = 0
            best_score = trial_score
            best_kwargs = kwargs
            smooth_score = trial_score
            smoothing = 0.05
            running = []
            since_save = 0
            since_new_best = 0
            while n_complete < optimizer.budget:
                # Add new jobs
                while (
                        len(running) < optimizer.num_workers
                        and len(running) + n_complete < optimizer.budget
                ):
                    candidate = optimizer.ask()
                    job = pool.apply_async(func=score_fn, args=candidate.args, kwds=candidate.kwargs)
                    running.append((candidate, job))
                # Collect finished jobs
                still_running = []
                for candidate, job in running:
                    if job.ready():
                        since_new_best += 1
                        result = job.get()
                        optimizer.tell(candidate, result)
                        if result < best_score:
                            best_score = result
                            best_kwargs = candidate.kwargs
                            since_new_best = 0
                        if not np.isnan(result) and not np.isinf(result):
                            smooth_score = smooth_score * (1 - smoothing) + result * smoothing
                        pbar.set_description(f'Best: {best_score:.4g}, Last: {result:.4g}, {since_new_best:,.0f} since best',
                                             refresh=False)
                        pbar.update()
                        n_complete += 1
                        since_save += 1
                        if save_after and since_save >= save_after:
                            with open('tmp.pkl', 'wb') as f:
                                pickle.dump(optimizer.provide_recommendation().kwargs, f)
                            with open('tmp_best.pkl', 'wb') as f:
                                pickle.dump(best_kwargs, f)
                            since_save = 0
                    else:
                        still_running.append((candidate, job))
                running = still_running
    except KeyboardInterrupt:
        LOG.info('Manually stopped (KeyboardInterrupt). Providing best recommendation with available data.')

    args, kwargs = optimizer.provide_recommendation().value
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(kwargs, f)
    with open('tmp_best.pkl', 'wb') as f:
        pickle.dump(best_kwargs, f)
    return kwargs, best_kwargs
