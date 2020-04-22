import logging

import numpy as np

LOG = logging.getLogger(__name__)


def odeint(func, y0, t, args=()):
    LOG.debug('Integrating %s over %s times and %s states', func, len(t), len(y0))
    y = y0
    out = [y]
    for t0, t1 in zip(t[:-1], t[1:]):
        dydt = np.array(func(y, t0, *args))
        y = y + dydt * (t1 - t0)
        out.append(y)
    out = np.array(out)
    LOG.debug('Shape of dydt is %s', dydt.shape)
    LOG.debug('Shape of output is %s', out.shape)
    return out
