import logging
import pickle

import nevergrad as ng
import numpy as np
import pymc3 as pm

import util
from data import (
    lombardia, )
from sde_seir import TimeDependentEulerMaruyama, sde_fn, default_param_priors, guessed_param_priors, \
    test_val_from_data
from seir import simple_obs_model, I_FIRST_OBS, SIM_TIME

logging.basicConfig(level=logging.DEBUG, format='(%(levelname).1s) %(name)s: %(message)s')

LOG = logging.getLogger(__name__)

guess = None
with open('fit.pkl', 'rb') as f:
    guess = pickle.load(f)[0]

LOG.info('Creating new SDE SEIR model with initial parameter guess: %s', guess)
with pm.Model() as model:

    if guess is None:
        params = default_param_priors()
    else:
        params = guessed_param_priors(guess)

    t = np.array(SIM_TIME)

    testval = test_val_from_data()

    LOG.debug('Constructing SDE node')
    states = TimeDependentEulerMaruyama(
        'state',
        t=np.array(SIM_TIME),
        sde_fn=sde_fn,
        sde_pars=params,
        testval=testval,
        shape=testval.shape
    )
    s, e, i0, i0d, i1, i2, f, fd, r, rd = [states[i, I_FIRST_OBS:] for i in range(10)]

    LOG.debug('Constructing observation models')
    # Observation models (keep it simple)
    simple_obs_model('c', i0d + i1 + i2 + fd + rd, lombardia['totale_casi'])
    simple_obs_model('f', fd, lombardia['deceduti'])
    simple_obs_model('i0d', i0d, lombardia['isolamento_domiciliare'])
    simple_obs_model('i1', i1, lombardia['ricoverati_con_sintomi'])
    simple_obs_model('i2', i2, lombardia['terapia_intensiva'])
    simple_obs_model('rd', rd, lombardia['dimessi_guariti'])

LOG.info('Generating nevergrad instrumentation from model')
instrum = {v.name: v.distribution.default() for v in model.free_RVs if v.name != 'state'}
print(instrum)
instrum = {k: ng.p.Scalar(init=v) for k, v in instrum.items()}
instrum['state'] = ng.p.Array(init=model['state'].distribution.default())
instrum = ng.p.Instrumentation(**instrum)


# nevergrad minimizes so we need to negate the logp to maximize it
# and since logp doesn't need the kwargs to be expanded let's just leave them as a dict
def score(**kwargs):
    return -model.logp(kwargs)
    # return -model.logp({k: np.exp(v) for k, v in kwargs.items()})


LOG.info('Maximizing model.logp with nevergrad')
rec, best = util.fit_nevergrad_model(instrum, 1_000, ng.optimizers.ParaPortfolio, score)

# LOG.info('Sampling model')
# with model:
#     trace = pm.sample(1_000, tune=1_000)
#
# LOG.debug('Exporting trace to dataframe')
# trace = pm.trace_to_dataframe(trace)
# LOG.debug('Saving trace dataframe')
# pd.to_pickle(trace, 'sde_trace.pkl')
