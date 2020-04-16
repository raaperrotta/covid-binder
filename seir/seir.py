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
rd cases are marked recovered (dimessi_guariti). TODO: Should this be just i1 and i2 recoveries?
We also have data for total tests administered (tamponi) and total patients hospitalized (totale_ospedalizzati).

For simplicity, we model the observation errors generically as Gaussian noise with constant plus linear scaling sigma.

Some thoughts:
- Knowing things like total mortality, we could determine recovery rate from mortality rate or vice versa.
- This also applies to fractions of mild, severe, and critical cases.
- And to typical duration of critical cases.
"""
import os
from collections import OrderedDict, namedtuple
from functools import partial
import multiprocessing as mp

import nevergrad as ng
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.integrate import odeint
import theano.tensor as tt
from theano.compile.ops import as_op
import theano
from tqdm import tqdm

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


SIM_TIME = to_sim_day(pd.date_range(DATE_OF_SIM_TIME_ZERO, lombardia['data'].max(), freq='1d'))
I_FIRST_OBS = int(to_sim_day(lombardia['data'].min()))

# These are switch points in our model at which times the interaction rate between people without detected cases changes
T1 = to_sim_day(DATE_OF_LOMBARDY_LOCKDOWN)
T2 = to_sim_day(DATE_OF_SHUTDOWN_OF_NONESSENTIALS)

Param = namedtuple('Param', ['min', 'max', 'default', 'description'])

PARAMS = OrderedDict(
    t0=Param(
        DATE_OF_SIM_TIME_ZERO + pd.to_timedelta('1d'),
        DATE_OF_LOMBARDY_LOCKDOWN - pd.to_timedelta('1d'),
        pd.to_datetime('1 Feb 2020'),
        'Date of first possible transmission',
    ),
    e0=Param(0, 100, 10, 'Exposed population as of t0'),
    beta0_a=Param(0, 2, 0.33, 'Transmission rate from unknown cases before 8 March'),
    beta0_b=Param(0, 1, 0.09, 'Transmission rate from unknown cases after 8 March'),
    beta0_c=Param(0, 1, 0.03, 'Transmission rate from unknown cases after 21 March'),
    beta0d=Param(0, 1, 0.05, 'Transmission rate from known cases with mild symptoms'),
    beta1=Param(0, 1, 0.05, 'Transmission rate from known cases with severe symptoms'),
    beta2=Param(0, 1, 0.01, 'Transmission rate from known cases with critical symptoms'),
    sigmae=Param(0, 1, 0.1, 'Progression rate for development of mild symptoms'),
    sigma0=Param(0, 1, 0.2, 'Progression rate for development of severe symptoms when undetected'),
    sigma0d=Param(0, 1, 0.01, 'Progression rate for development of severe symptoms when detected'),
    sigma1=Param(0, 1, 0.02, 'Progression rate for development of critical symptoms'),
    theta=Param(0, 0.5, 0.1, 'Rate of testing for undetected cases with mild symptoms'),
    mu0=Param(0, 0.1, 0.01, 'Fatality rate for undetected mild cases'),
    mu0d=Param(0, 0.1, 0.01, 'Fatality rate for detected mild cases'),
    mu1=Param(0, 0.1, 0.01, 'Fatality rate for severe cases'),
    mu2=Param(0, 0.1, 0.006, 'Fatality rate for critical cases'),
    gamma0=Param(0, 0.1, 0.015, 'Recovery rate for undetected mild cases'),
    gamma0d=Param(0, 0.1, 0.01, 'Recovery rate for detected mild cases'),
    gamma1=Param(0, 0.1, 0.03, 'Recovery rate for severe cases'),
    gamma2=Param(0, 0.5, 0.1, 'Recovery rate for critical cases'),
)


def naive_ternary(conditional, if_true, if_false):
    return if_false + (if_true - if_false) * conditional


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

    # # If data for recovered individuals includes those that recovered at home
    # dr = recovered_mild_undetected
    # drd = recovered_mild_detected + recovered_severe + recovered_critical

    # If data for recovered individuals does NOT include those that recovered at home
    dr = recovered_mild_undetected + recovered_mild_detected
    drd = recovered_severe + recovered_critical

    return ds, de, di0, di0d, di1, di2, df, dfd, dr, drd


def run_odeint(t, e0, t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0,
               mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2, convert_times=True):

    if convert_times:
        t = to_sim_day(pd.to_datetime(t))
        t0 = to_sim_day(pd.to_datetime(t0))

    state0 = POPULATION_OF_LOMBARDY - e0, e0, 0, 0, 0, 0, 0, 0, 0, 0
    params = (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d, sigma1, theta, mu0, mu0d,
              mu1, mu2, gamma0, gamma0d, gamma1, gamma2)

    states = odeint(
        partial(seir, ternary=util.ternary),
        t=t,
        y0=state0,
        args=(params,),
    )
    return [y for y in states.T]  # one array per state


def simple_obs_model(name, simulated, observed):
    obs_sd0 = pm.Exponential(name + '_obs_sd0', 100.0)
    obs_sd1 = pm.Exponential(name + '_obs_sd1', 0.1)
    pm.Normal(name + '_obs', simulated, obs_sd0 + obs_sd1 * simulated, observed=observed)


# @as_op(itypes=[tt.dscalar] * 21, otypes=[tt.dvector])
# def custom_ode_op(e0, t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d,
#                   sigma1, theta, mu0, mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2):
#     s0 = POPULATION_OF_LOMBARDY - e0
#     state0 = s0, e0, 0, 0, 0, 0, 0, 0, 0, 0
#     params = (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d,
#               sigma1, theta, mu0, mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2)
#     return odeint(
#         seir,
#         # partial(seir, ternary=util.ternary),
#         t=SIM_TIME,
#         y0=state0,
#         args=(params,),
#     )

@as_op(itypes=[tt.dvector, tt.dvector], otypes=[tt.dmatrix])
def custom_ode_op(state0, params):
    return odeint(
        partial(seir, ternary=util.ternary),
        t=SIM_TIME,
        y0=state0,
        args=(params,),
    )


def create_model():

    with pm.Model() as model:

        # Transmission rates
        beta0_a = pm.Lognormal('beta0_a', mu=0.33, sd=1)
        beta0_b = pm.Lognormal('beta0_b', mu=0.07, sd=1)
        beta0_c = pm.Lognormal('beta0_c', mu=0.05, sd=1)
        beta0d = pm.Lognormal('beta0d', mu=0.15, sd=1)
        beta1 = pm.Lognormal('beta1', mu=0.02, sd=1)
        beta2 = pm.Lognormal('beta2', mu=0.05, sd=1)

        # Progression rates
        sigmae = pm.Lognormal('sigmae', mu=np.log(0.2), sd=1)
        sigma0 = pm.Lognormal('sigma0', mu=np.log(0.05), sd=1)
        sigma0d = pm.Lognormal('sigma0d', mu=np.log(0.04), sd=1)
        sigma1 = pm.Lognormal('sigma1', mu=np.log(0.02), sd=1)

        # Recovery rates
        gamma0 = pm.Lognormal('gamma0', mu=np.log(0.01), sd=1)
        gamma0d = pm.Lognormal('gamma0d', mu=np.log(0.01), sd=1)
        gamma1 = pm.Lognormal('gamma1', mu=np.log(0.01), sd=1)
        gamma2 = pm.Lognormal('gamma2', mu=np.log(0.01), sd=1)

        # Fatality rates
        mu0 = pm.Lognormal('mu0', mu=np.log(0.0007), sd=1)
        mu0d = pm.Lognormal('mu0d', mu=np.log(0.0006), sd=1)
        mu1 = pm.Lognormal('mu1', mu=np.log(0.003), sd=1)
        mu2 = pm.Lognormal('mu2', mu=np.log(0.005), sd=1)

        # Testing rate
        theta = pm.Lognormal('theta', mu=np.log(0.002), sd=1)

        # Initial conditions
        t0 = pm.Uniform('t0', to_sim_day(pd.to_datetime('24 Jan 2020')),
                        to_sim_day(pd.to_datetime('14 Feb 2020')))
        e0 = pm.Lognormal('e0', np.log(100), 2)
        s0 = pm.Deterministic('s0', POPULATION_OF_LOMBARDY - e0)

        # state0 = s0, e0, 0, 0, 0, 0, 0, 0, 0, 0
        # params = (t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d,
        #           sigma1, theta, mu0, mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2)
        #
        # # Dynamics model
        ode = pm.ode.DifferentialEquation(func=seir, t0=0, times=SIM_TIME, n_states=len(state0), n_theta=len(params))
        # states = ode(state0, params)

        state0 = pm.math.concatenate([s0, e0, 0, 0, 0, 0, 0, 0, 0, 0])
        params = pm.math.concatenate([t0, beta0_a, beta0_b, beta0_c, beta0d, beta1, beta2, sigmae, sigma0, sigma0d,
                                      sigma1, theta, mu0, mu0d, mu1, mu2, gamma0, gamma0d, gamma1, gamma2])
        # Dynamics model
        states = custom_ode_op(state0, params)

        s, e, i0, i0d, i1, i2, f, fd, r, rd = [states[:, i] for i in range(len(state0))]

        # Observation models (keep it simple)
        simple_obs_model('c', (i0d + i1 + i2 + fd + rd)[I_FIRST_OBS:], lombardia['totale_casi'])
        simple_obs_model('f', fd[I_FIRST_OBS:], lombardia['deceduti'])
        simple_obs_model('i0d', i0d[I_FIRST_OBS:], lombardia['isolamento_domiciliare'])
        simple_obs_model('i1', i1[I_FIRST_OBS:], lombardia['ricoverati_con_sintomi'])
        simple_obs_model('i2', i2[I_FIRST_OBS:], lombardia['terapia_intensiva'])
        simple_obs_model('rd', rd[I_FIRST_OBS:], lombardia['dimessi_guariti'])

        return model


def create_model_from_guess(best_guess):

    # Get rid of any zeros
    best_guess = {k: v or 1e-3 for k, v in best_guess.items()}

    with pm.Model() as model:

        priors = {k: pm.Lognormal(k, mu=v, sd=0.6, testval=v) for k, v in best_guess.items()}

        s0 = pm.Deterministic('s0', POPULATION_OF_LOMBARDY - priors['e0'])

        # state0 = s0, priors['e0'], 0, 0, 0, 0, 0, 0, 0, 0
        # params = [priors[k] for k in ('t0', 'beta0_a', 'beta0_b', 'beta0_c', 'beta0d', 'beta1', 'beta2', 'sigmae',
        #                               'sigma0', 'sigma0d', 'sigma1', 'theta', 'mu0', 'mu0d', 'mu1', 'mu2', 'gamma0',
        #                               'gamma0d', 'gamma1', 'gamma2')]
        # # Dynamics model
        # ode = pm.ode.DifferentialEquation(func=seir, t0=0, times=SIM_TIME, n_states=len(state0), n_theta=len(params))
        # states = ode(state0, params)

        state0 = pm.math.stack([s0, priors['e0'], 0, 0, 0, 0, 0, 0, 0, 0])
        params = pm.math.stack([priors[k] for k in ('t0', 'beta0_a', 'beta0_b', 'beta0_c', 'beta0d', 'beta1', 'beta2', 'sigmae',
                                      'sigma0', 'sigma0d', 'sigma1', 'theta', 'mu0', 'mu0d', 'mu1', 'mu2', 'gamma0',
                                      'gamma0d', 'gamma1', 'gamma2')])
        # Dynamics model
        states = custom_ode_op(state0, params)

        s, e, i0, i0d, i1, i2, f, fd, r, rd = [states[:, i] for i in range(10)]

        # Observation models (keep it simple)
        simple_obs_model('c', (i0d + i1 + i2 + fd + rd)[I_FIRST_OBS:], lombardia['totale_casi'])
        simple_obs_model('f', fd[I_FIRST_OBS:], lombardia['deceduti'])
        simple_obs_model('i0d', i0d[I_FIRST_OBS:], lombardia['isolamento_domiciliare'])
        simple_obs_model('i1', i1[I_FIRST_OBS:], lombardia['ricoverati_con_sintomi'])
        simple_obs_model('i2', i2[I_FIRST_OBS:], lombardia['terapia_intensiva'])
        simple_obs_model('rd', rd[I_FIRST_OBS:], lombardia['dimessi_guariti'])

        return model


def compute_trace_score(simulated, observed):
    simulated = np.log(1 + simulated)
    observed = np.log(1 + observed)
    diff = (simulated - observed)
    diff2 = diff * diff
    return sum(diff2)


def compute_overall_score(**kwargs):
    """Compute goodness-of-fit score suitable for nevergrad optimization"""
    s, e, i0, i0d, i1, i2, f, fd, r, rd = run_odeint(SIM_TIME, convert_times=False, **kwargs)
    return (
        compute_trace_score((i0d + i1 + i2 + fd + rd)[I_FIRST_OBS:], lombardia['totale_casi']) +
        compute_trace_score(fd[I_FIRST_OBS:], lombardia['deceduti']) +
        compute_trace_score(i0d[I_FIRST_OBS:], lombardia['isolamento_domiciliare']) +
        compute_trace_score(i1[I_FIRST_OBS:], lombardia['ricoverati_con_sintomi']) +
        compute_trace_score(i2[I_FIRST_OBS:], lombardia['terapia_intensiva']) +
        compute_trace_score(rd[I_FIRST_OBS:], lombardia['dimessi_guariti'])
    )


def fit_nevergrad_model(budget=12_000):

    parameters = {
        name:
            ng.p.Scalar(lower=param.min, upper=param.max, init=param.default)
            if isinstance(param.default, (int, float)) else
            ng.p.Scalar(lower=to_sim_day(param.min), upper=to_sim_day(param.max), init=to_sim_day(param.default))
        for name, param in PARAMS.items()
    }

    instrumentation = ng.p.Instrumentation(**parameters)

    # Run the score function once to catch bugs before dispatching to the multiprocessing Pool
    # That makes debugging much easier.
    args, kwargs = instrumentation.value
    trial_score = compute_overall_score(**kwargs)

    # optimizer = ng.optimizers.TwoPointsDE(
    #     instrumentation, budget=budget, num_workers=1,
    # )
    # optimizer.minimize(compute_overall_score)

    num_processes = os.cpu_count()
    optimizer = ng.optimizers.ParaPortfolio(
        instrumentation, budget=budget, num_workers=num_processes,
    )
    with mp.Pool(processes=num_processes) as pool, tqdm(total=budget) as pbar:
        n_complete = 0
        best_score = trial_score
        best_kwargs = kwargs
        smooth_score = trial_score
        smoothing = 0.3
        running = []
        while n_complete < optimizer.budget:
            # Add new jobs
            while (
                    len(running) < optimizer.num_workers
                    and len(running) + n_complete < optimizer.budget
            ):
                candidate = optimizer.ask()
                job = pool.apply_async(func=compute_overall_score, args=candidate.args, kwds=candidate.kwargs)
                running.append((candidate, job))
            # Collect finished jobs
            still_running = []
            for candidate, job in running:
                if job.ready():
                    result = job.get()
                    optimizer.tell(candidate, result)
                    if result < best_score:
                        best_score = result
                        best_kwargs = candidate.kwargs
                    smooth_score = smooth_score * (1 - smoothing) + result * smoothing
                    pbar.set_description(f'Best: {best_score:.4g}, Smooth: {smooth_score:.4g}, Last: {result:.4g}',
                                         refresh=False)
                    pbar.update()
                    n_complete += 1
                else:
                    still_running.append((candidate, job))
            running = still_running

    args, kwargs = optimizer.provide_recommendation().value
    return kwargs, best_kwargs
