"""SEIR model for COVID-19

This file defines the differential equation underlying a SEIR model of COVID-19.

The model has the following compartments:
- s: susceptible
- e: exposed
- i0: infectious with mild symptoms
- i0d: i0 patients with confirmed cases
- i1: infectious with severe symptoms (always detected)
- i2: infectious in critical condition (always detected)
- f: fatalities that went undetected
- fd: fatalities from detected cases
- r: recovered patients that went undetected
- rd: recovered patients from detected cases

The model makes the following assumptions:
- No re-susceptibility. Recovered individuals stay that way.
- No-one is tested until they are infectious.
- i0 patients recover at home
- i1 patients are admitted to the hospital
- i2 patients are treated in ICU
- i1 and i2 patients are always detected

Transitions in the model happen according to the following:
s->e: susceptible patients are exposed by coming into contact with infectious patients
  s / n * (beta0 * i0 + beta0d * i0d + beta1 * i1 + beta2 * i2)
e->i0: exposed individuals develop mild symptoms and become infectious
  e * sigmae
i0->i0d: some individuals with mild symptoms are tested and their condition is detected
  i0 * theta
i0->i1: symptoms can progress from mild to severe
  i0 * sigma0
i0d->i1: progression for detected cases can be different
  i0d * sigma0d
i1->i2: symptoms can progress from severe to critical
  i1 * sigma1
i0->r: undetected patients can recover
  i0 * gamma0
i_->rd: likewise detected cases
  i_ * gamma_
i0->f: as with recovery, undetected cases can result in undetected fatalities (are they tested after death?)
  i0 * mu0
i_->fd: likewise detected cases
  i_ * mu_

Parameters used above:
- n: the total population size, which is fixed in our model
- beta_: the rate of transmission from patients in the corresponding group
  beta_ = R0 * lam_: The transmissibility, R0, times the interaction constant lam_
- theta: the rate of detection. Incorporates rate of testing and rate of true positives.
- sigma_: the progression factors
- gamma_: the recovery factors
- mu_: the fatality factors

Our goal is to fit this model to the data from Lombardy, Italy.
Confirmed cases (totale_casi) are the sum of i0d, i1, i2, fd, and rd.
Fatalities (dimessi_guariti) are fd.
i0d cases are put in home isolation (isolamento_domiciliare).
i1 cases are admitted into the hospital (ricoverati_con_sintomi).
i2 cases are sent to ICU (terapia_intensiva).
rd cases are marked recovered (dimessi_guariti).
We also have data for total tests administered (tamponi) and total patients hospitalized (totale_ospedalizzati).

For simplicity, we model the observation errors generically as Gaussian noise with constant plus linear scaling sigma.

Some thoughts:
- Knowing things like total mortality, we could determine recovery rate from mortality rate or vice versa.
- This also applies to fractions of mild, severe, and critical cases.
- And to typical duration of critical cases.
"""

from collections import OrderedDict, namedtuple
from functools import partial

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.integrate import odeint

import util
from data import (
    lombardia,
    POPULATION_OF_LOMBARDY,
    DATE_OF_LOMBARDY_LOCKDOWN,
    DATE_OF_SHUTDOWN_OF_NONESSENTIALS,
)

DATE_OF_SIM_TIME_ZERO = pd.to_datetime('1 Jan 2020')


def to_sim_day(date):
    return np.array((date - DATE_OF_SIM_TIME_ZERO).days, dtype=float)


# These are switch points in our model at which times the interaction rate between people without detected cases changes
T1 = to_sim_day(DATE_OF_LOMBARDY_LOCKDOWN)
T2 = to_sim_day(DATE_OF_SHUTDOWN_OF_NONESSENTIALS)

Param = namedtuple('Param', ['min', 'max', 'default', 'description'])

PARAMS = OrderedDict(
    t0=Param(
        DATE_OF_SIM_TIME_ZERO + pd.to_timedelta('1d'),
        DATE_OF_LOMBARDY_LOCKDOWN - pd.to_timedelta('1d'),
        pd.to_datetime('2 Feb 2020'),
        'Date of first possible transmission',
    ),
    e0=Param(0, 1e4, 1e3, 'Exposed population as of t0'),
    beta0_a=Param(0, 2, 0.5, 'Transmission rate from unknown cases before 8 March'),
    beta0_b=Param(0, 1, 0.15, 'Transmission rate from unknown cases after 8 March'),
    beta0_c=Param(0, 1, 0.05, 'Transmission rate from unknown cases after 21 March'),
    beta0d=Param(0, 1, 0.15, 'Transmission rate from known cases with mild symptoms'),
    beta1=Param(0, 1, 0.05, 'Transmission rate from known cases with severe symptoms'),
    beta2=Param(0, 1, 0.05, 'Transmission rate from known cases with critical symptoms'),
    sigmae=Param(0, 1, 1/5.1, 'Progression rate for development of mild symptoms'),
    sigma0=Param(0, 1, 1/6, 'Progression rate for development of severe symptoms when undetected'),
    sigma0d=Param(0, 1, 1/6, 'Progression rate for development of severe symptoms when detected'),
    sigma1=Param(0, 1, 1/4.5, 'Progression rate for development of critical symptoms'),
    theta=Param(0, 0.1, 0.002, 'Rate of testing for undetected cases with mild symptoms'),
    mu0=Param(0, 0.01, 0.0004, 'Fatality rate for undetected mild cases'),
    mu0d=Param(0, 0.01, 0.0003, 'Fatality rate for detected mild cases'),
    mu1=Param(0, 0.01, 0.001, 'Fatality rate for severe cases'),
    mu2=Param(0, 0.01, 0.002, 'Fatality rate for critical cases'),
    gamma0=Param(0, 0.01, 0.001, 'Recovery rate for undetected mild cases'),
    gamma0d=Param(0, 0.01, 0.001, 'Recovery rate for detected mild cases'),
    gamma1=Param(0, 0.01, 0.001, 'Recovery rate for severe cases'),
    gamma2=Param(0, 0.01, 0.001, 'Recovery rate for critical cases'),
)


def seir(state, t, params, ternary=pm.math.switch):
    """Compute the derivatives of our SEIR model

    The optional argument ternary allows us to re-use this function outside of a pymc model context by supplying the
    generic python ternary function defined in util.py in place of the pymc switch operator.
    """
    # Split the state and parameter vectors for more readable code
    # We must slice the vectors to make the length explicit to support of pymc's deferred execution
    s, e, i0, i0d, i1, i2, f, fd, r, rd = [state[i] for i in range(10)]
    (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0, mu0d, mu1, mu2,
     gamma0, gamma0d, gamma1, gamma2) = [params[i] for i in range(20)]

    beta0 = ternary(t <= t0, 0, ternary(t <= T1, beta0_a, ternary(t <= T2, beta0_b, beta0_c)))
    sigmae = ternary(t <= t0, 0, sigmae)  # freeze the simulation before time t0 by keeping beta0 and sigmae at zero

    newly_exposed = (beta0 * i0 + beta0d * i0d + beta1 * i1 + beta2 * i2) * s / POPULATION_OF_LOMBARDY
    developed_mild_symptoms = sigmae * e
    tested_with_mild_symptoms = i0 * theta
    symptoms_mild_to_severe_undetected = i0 * sigma0
    symptoms_mild_to_severe_detected = i0d * sigma0d
    symptoms_severe_to_critical = i1 * sigma1
    recovered_mild_undetected = i0 * gamma0
    recovered_mild_detected = i0d * gamma0d
    recovered_severe = i1 * gamma1
    recovered_critical = i2 * gamma2
    died_mild_undetected = i0 * mu0
    died_mild_detected = i0d * mu0d
    died_severe = i1 * mu1
    died_critical = i2 * mu2

    ds = -newly_exposed
    de = newly_exposed - developed_mild_symptoms
    di0 = (developed_mild_symptoms - tested_with_mild_symptoms - symptoms_mild_to_severe_undetected -
           recovered_mild_undetected - died_mild_undetected)
    di0d = tested_with_mild_symptoms - symptoms_mild_to_severe_detected - recovered_mild_detected - died_mild_detected
    di1 = (symptoms_mild_to_severe_undetected + symptoms_mild_to_severe_detected - symptoms_severe_to_critical -
           recovered_severe - died_severe)
    di2 = symptoms_severe_to_critical - recovered_critical - died_critical
    df = died_mild_undetected
    dfd = died_mild_detected + died_critical + died_severe
    dr = recovered_mild_undetected
    drd = recovered_mild_detected + recovered_severe + recovered_critical

    return ds, de, di0, di0d, di1, di2, df, dfd, dr, drd


def run_odeint(t, e0, t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0,
               mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2):

    t = to_sim_day(pd.to_datetime(t))
    t0 = to_sim_day(pd.to_datetime(t0))

    state0 = POPULATION_OF_LOMBARDY - e0, e0, 0, 0, 0, 0, 0, 0, 0, 0
    params = (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0, mu0d,
              mu1, mu2, gamma0, gamma0d, gamma1, gamma2)

    states = odeint(
        partial(seir, ternary=util.ternary),
        t=t,
        y0=state0,
        args=(params,)
    )
    return [y for y in states.T]  # one array per state


def simple_obs_model(name, simulated, observed):
    obs_sd0 = pm.Exponential(name + '_obs_sd0', 10.0)
    obs_sd1 = pm.Exponential(name + '_obs_sd1', 0.1)
    pm.Normal(name + '_obs', simulated, obs_sd0 + obs_sd1 * simulated, observed=observed)


def fit(**new_kwargs):

    t = to_sim_day(pd.date_range(DATE_OF_SIM_TIME_ZERO, lombardia['data'].max(), freq='1d'))
    i_first_obs = int(to_sim_day(lombardia['data'].min()))

    with pm.Model():
        beta0_a = pm.Lognormal('beta0_a', mu=0.5, sd=0.5, testval=0.5)
        beta0_b = pm.Lognormal('beta0_b', mu=0.5, sd=0.5, testval=0.5)
        beta0_c = pm.Lognormal('beta0_c', mu=0.5, sd=0.5, testval=0.5)
        beta0d = pm.Lognormal('beta0d', mu=0.17, sd=0.5)
        beta1 = pm.Lognormal('beta1', mu=0.17, sd=0.5)
        beta2 = pm.Lognormal('beta2', mu=0.02, sd=0.5)

        sigmae = pm.Beta('sigmae', mu=0.2, sd=0.02)
        sigma0 = pm.Beta('sigma0', mu=0.2, sd=0.02)
        sigma0d = pm.Beta('sigma0d', mu=0.2, sd=0.02)
        sigma1 = pm.Beta('sigma1', mu=0.2, sd=0.02)

        gamma0 = pm.Beta('gamma0', mu=0.08, sigma=0.02)
        gamma0d = pm.Beta('gamma0d', mu=0.08, sigma=0.02)
        gamma1 = pm.Beta('gamma1', mu=0.08, sigma=0.02)
        gamma2 = pm.Beta('gamma2', mu=0.08, sigma=0.02)

        mu0 = pm.Beta('mu0', mu=0.0004, sd=0.0002)
        mu0d = pm.Beta('mu0d', mu=0.0004, sd=0.0002)
        mu1 = pm.Beta('mu1', mu=0.0004, sd=0.0002)
        mu2 = pm.Beta('mu2', mu=0.0004, sd=0.0002)

        theta = pm.Uniform('theta', 0.0, 1.0, testval=0.004)

        t0 = pm.Uniform('t0', to_sim_day(pd.to_datetime('14 Jan 2020')), to_sim_day(pd.to_datetime('1 Feb 2020')))

        e0 = pm.Lognormal('e0', np.log(1000), 2)
        s0 = pm.Deterministic('s0', POPULATION_OF_LOMBARDY - e0)

        state0 = s0, e0, 0, 0, 0, 0, 0, 0, 0, 0
        params = (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0,
                  mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2)

        # Dynamics model
        ode = pm.ode.DifferentialEquation(func=seir, t0=0, times=t, n_states=len(state0), n_theta=len(params))
        states = ode(state0, params)
        s, e, i0, i0d, i1, i2, f, fd, r, rd = [states[:, i] for i in range(len(state0))]

        # Observation models (keep it simple)
        simple_obs_model('c', (i0d + i1 + i2 + fd + rd)[i_first_obs:], lombardia['totale_casi'])
        simple_obs_model('f', fd[i_first_obs:], lombardia['deceduti'])
        simple_obs_model('i0d', i0d[i_first_obs:], lombardia['isolamento_domiciliare'])
        simple_obs_model('i1', i1[i_first_obs:], lombardia['ricoverati_con_sintomi'])
        simple_obs_model('i2', i2[i_first_obs:], lombardia['terapia_intensiva'])
        simple_obs_model('rd', rd[i_first_obs:], lombardia['dimessi_guariti'])

        kwargs = dict(draws=100, tune=100, target_accept=0.234, compute_convergence_checks=False)
        kwargs.update(new_kwargs)
        trace = pm.sample(**kwargs)
        return trace
