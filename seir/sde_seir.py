import logging
import pickle
from functools import partial

import nevergrad as ng
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

import euler
from data import (
    lombardia, POPULATION_OF_LOMBARDY,
)
from seir import simple_obs_model, I_FIRST_OBS, SIM_TIME, seir
import util

LOG = logging.getLogger(__name__)


class TimeDependentEulerMaruyama(pm.distributions.distribution.Continuous):
    R"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    t : float array
        time of discretization, must be monotonically increasing
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as ``*args`` to ``sde_fn``
    """

    def __init__(self, t, sde_fn, sde_pars, *args, **kwds):
        super().__init__(*args, **kwds)
        self.t0 = tt.as_tensor_variable(t[:-1])
        self.dt = tt.as_tensor_variable(np.diff(t))
        self.sqrt_dt = np.sqrt(self.dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        x0 = x[:, :-1]
        f, g = self.sde_fn(x0, self.t0, self.sde_pars)
        mu = x0 + self.dt * f
        sd = self.sqrt_dt * g
        return tt.sum(pm.Normal.dist(mu=mu, sigma=sd).logp(x[:, 1:]))


def sde_fn(y, t, p):
    return seir(y, t, p), 100 + y * 0.1


def default_param_priors():
    LOG.debug('Generating default Lognormal priors for model parameters')
    # Transmission rates
    beta0_a = pm.Lognormal('beta0_a', mu=np.log(0.33), sd=1)
    beta0_b = pm.Lognormal('beta0_b', mu=np.log(0.07), sd=1)
    beta0_c = pm.Lognormal('beta0_c', mu=np.log(0.05), sd=1)
    beta0d = pm.Lognormal('beta0d', mu=np.log(0.15), sd=1)
    beta1 = pm.Lognormal('beta1', mu=np.log(0.02), sd=1)
    beta2 = pm.Lognormal('beta2', mu=np.log(0.05), sd=1)
    # Progression rates
    sigmae = pm.Lognormal('sigmae', mu=np.log(0.2), sd=1)
    sigma0 = pm.Lognormal('sigma0', mu=np.log(0.1), sd=2)
    sigma0d = pm.Lognormal('sigma0d', mu=np.log(0.1), sd=2)
    sigma1 = pm.Lognormal('sigma1', mu=np.log(0.1), sd=2)
    # Recovery rates
    gamma0 = pm.Lognormal('gamma0', mu=np.log(0.1), sd=2)
    gamma0d = pm.Lognormal('gamma0d', mu=np.log(0.1), sd=2)
    gamma1 = pm.Lognormal('gamma1', mu=np.log(0.1), sd=2)
    gamma2 = pm.Lognormal('gamma2', mu=np.log(0.1), sd=2)
    # Fatality rates
    mu0 = pm.Lognormal('mu0', mu=np.log(0.001), sd=2)
    mu0d = pm.Lognormal('mu0d', mu=np.log(0.001), sd=2)
    mu1 = pm.Lognormal('mu1', mu=np.log(0.003), sd=2)
    mu2 = pm.Lognormal('mu2', mu=np.log(0.005), sd=2)
    # Testing rate
    theta = pm.Lognormal('theta', mu=np.log(0.002), sd=2)
    return (beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d,
            sigma1, theta, mu0, mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2)


def guessed_param_priors(guess):
    sd = 0.2
    nonzero = 0.00001
    LOG.debug('Generating Lognormal priors for guessed values with sd=%f and %f in place of zeros.', sd, nonzero)
    return [
        pm.Lognormal(k, mu=np.log(guess[k] or nonzero), sd=sd, testval=guess[k] or nonzero)
        for k in ('beta0_a', 'beta0_b', 'beta0_c', 'beta0d', 'beta1', 'beta2', 'sigmae',
                  'sigma0', 'sigma0d', 'sigma1', 'theta', 'mu0', 'mu0d', 'mu1', 'mu2', 'gamma0',
                  'gamma0d', 'gamma1', 'gamma2')
    ]


def test_val_from_data():
    LOG.debug('Creating SDE testval from data')
    testval = np.zeros((10, len(SIM_TIME)))
    # Fill in data values as initial guess for true states
    testval[3, I_FIRST_OBS:] = lombardia['isolamento_domiciliare']
    testval[4, I_FIRST_OBS:] = lombardia['ricoverati_con_sintomi']
    testval[5, I_FIRST_OBS:] = lombardia['terapia_intensiva']
    testval[7, I_FIRST_OBS:] = lombardia['deceduti']
    testval[9, I_FIRST_OBS:] = lombardia['dimessi_guariti']
    # Fill in known values as proxy for unknowns
    testval[1, I_FIRST_OBS:] = lombardia['isolamento_domiciliare']
    testval[2, I_FIRST_OBS:] = lombardia['isolamento_domiciliare']
    # Fill in times before data with first value (could use exp decay if needed)
    # Index with list [I_FIRST_OBS] so expression is a slice and keeps dimension
    testval[:, :I_FIRST_OBS] = testval[:, [I_FIRST_OBS]]
    # Compute susceptible as remainder of population
    testval[0, :] = POPULATION_OF_LOMBARDY - testval.sum(axis=0, keepdims=True)
    return testval


def test_val_from_euler(ts, p):
    LOG.debug('Creating SDE testval from euler integration')
    y0 = np.array([None, 0.671625219974568, 0.8309875445376148, 0, 0, 0, 0, 0, 0, 0])
    y0[0] = POPULATION_OF_LOMBARDY - sum(y0[1:])
    return euler.odeint(partial(seir, ternary=util.ternary), y0, ts, [p])


def create_model(guess=None) -> pm.Model:
    LOG.info('Creating new SDE SEIR model with initial parameter guess: %s', guess)
    with pm.Model() as model:

        if guess is None:
            params = default_param_priors()
            guess = {p.name: p.random(size=1_000).mean() for p in params}
        else:
            params = guessed_param_priors(guess)
        guess = [
            guess[k]
            for k in ('beta0_a', 'beta0_b', 'beta0_c', 'beta0d', 'beta1', 'beta2', 'sigmae',
                      'sigma0', 'sigma0d', 'sigma1', 'theta', 'mu0', 'mu0d', 'mu1', 'mu2', 'gamma0',
                      'gamma0d', 'gamma1', 'gamma2')
        ]

        t = np.array(SIM_TIME)

        # testval = test_val_from_data()
        testval = test_val_from_euler(t, guess)

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

        return model


def nevergrad_fit(model):
    LOG.info('Generating nevergrad instrumentation from model')
    instrum = ng.p.Instrumentation(**{v.name: ng.p.Array(
        init=v.distribution.default()
    ) for v in model.free_RVs})

    # nevergrad minimizes so we need to negate the logp to maximize it
    # and since logp doesn't need the kwargs to be expanded let's just leave them as a dict
    def score(**kwargs):
        return -model.logp(kwargs)

    LOG.info('Maximizing model.logp with nevergrad')
    return util.fit_nevergrad_model(instrum, 1_000, ng.optimizers.ParaPortfolio, score)


def main():
    with open('fit.pkl', 'rb') as f:
        guess = pickle.load(f)[0]
    with create_model(guess) as model:
        # print(nevergrad_fit(model))
        LOG.info('Sampling model')
        trace = pm.sample(200, tune=200, cores=8, chains=6)
    LOG.debug('Exporting trace to dataframe')
    trace = pm.trace_to_dataframe(trace)
    LOG.debug('Saving trace dataframe')
    pd.to_pickle(trace, 'sde_trace.pkl')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='(%(levelname).1s) %(name)s: %(message)s')
    main()
