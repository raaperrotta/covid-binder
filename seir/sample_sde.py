import pickle

import nevergrad as ng
import numpy as np
import pandas as pd
import pymc3 as pm

from seir import util


regioni = pd.read_csv(
    'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
)
lombardia = regioni[regioni['denominazione_regione'] == 'Lombardia'].copy()
lombardia['data'] = pd.to_datetime(lombardia['data'].dt.date)  # Drop the time
lombardia.sort_values('data', inplace=True)

severity = pd.read_csv('severity.csv', parse_dates=['day'])
deaths = pd.read_csv('deaths.csv', parse_dates=['day'])
lethality = pd.read_csv('lethality.csv', parse_dates=['day'])
cases_by_age = pd.read_csv('cases.csv', parse_dates=['day'])

# Susceptible, exposed, plus 5 levels of infectious, the first of which is asymptomatic
N_INERT_STATES = 2
N_HIDDEN_STATES = 1
N_LETHAL_STATES = (
    3
)  # any state with symptoms is considered lethal (even if lethality is very low)
N_STATES = N_INERT_STATES + N_HIDDEN_STATES + N_LETHAL_STATES
N_AGES = 3  # Number of age groups

POPULATION = 10_060_574

AGE_DIST = np.array([0.5, 0.4, 0.1])
AGE_POP = POPULATION * AGE_DIST

N_PER_DAY = 4  # must divide 24 evenly
DT = 1 / N_PER_DAY
T = pd.date_range('8 Feb 2020', '15 May 2020', freq=f'{24 // N_PER_DAY}h')
N_T = len(T)

ERA_STARTS = pd.to_datetime(['8 Mar 2020', '21 Mar 2020'])
N_ERAS = len(ERA_STARTS) + 1
ERA_INDICES = np.sum(np.array(T) > ERA_STARTS[:, None], axis=0) if N_ERAS > 1 else [0] * N_T

I_FIRST = np.where(T == lombardia['data'].min())[0][0]
I_LAST = np.where(T == lombardia['data'].max())[0][0]

# Compute indices for alignment of stats with T
idx_time = pd.Series(range(N_T), T)

idx_severity = tuple(idx_time.asof(severity['day']).astype(int))
severity['pauci-lieve'] = severity['pauci-sintomaitco'] + severity['lieve']
# Distribute unspecified severity in proportion to specified data.
# This deserves more scrutiny. Is there a better way to handle the unspecified cases?
data_severity = np.array(severity[['asintomatico', 'pauci-lieve', 'severo', 'critico']]).T
data_severity *= 100 / (100 - severity['sintomatico non specificato'])

idx_deaths = tuple(idx_time.asof(deaths['day']).astype(int))
deaths['0-49'] = (
        deaths['0-9'] + deaths['10-19'] + deaths['20-29'] + deaths['30-39'] + deaths['40-49']
)
deaths['50-69'] = deaths['50-59'] + deaths['60-69']
deaths['>=70'] = deaths['70-79'] + deaths['80-89'] + deaths['>=90']
data_deaths = np.array(deaths[['0-49', '50-59', '>=70']]).T

idx_lethality = tuple(idx_time.asof(lethality['day']).astype(int))
# lethality can't just be added, we need to recompute it for our age groups
cases = (100 * deaths.set_index('day') / lethality.set_index('day')).fillna(0)
cases['0-49'] = (
        cases['0-9'] + cases['10-19'] + cases['20-29'] + cases['30-39'] + cases['40-49']
)
cases['50-59'] = cases['50-59'] + cases['60-69']
cases['>=70'] = cases['70-79'] + cases['80-89'] + cases['>=90']
data_lethality = 100 * data_deaths / np.array(cases[['0-49', '50-59', '>=70']]).T

# These age groups don't align perfectly with the others but they are close.
# Should we adjust for that?
idx_cases = tuple(idx_time.asof(cases_by_age['day']).astype(int))
cases_by_age['0-50'] = cases_by_age['0-18'] + cases_by_age['19-50']
data_cases = np.array(cases_by_age[['0-50', '51-70', '>70']]).T

with open('seir/tmp.pkl', 'rb') as f:
    kwargs = pickle.load(f)

print({k: v.shape for k, v in kwargs.items()})


def deterministic_ode(
        e0,
        beta,
        sigma,
        theta,
        gamma,
        mu
):

    # All living, non-recovered individuals in states above exposed can pass the virus
    # beta does not depend on age
    beta = beta[..., ERA_INDICES[:-1]]

    # All states except susceptible and critical can progress
    # Progression depends on age and detection status
    sigma = np.concatenate(
        (sigma, np.zeros((N_STATES - 2, N_AGES, 2, 1, 1, 1))), axis=3
    )  # recovered can't progress
    sigma = np.concatenate(
        (sigma, np.zeros((N_STATES - 2, N_AGES, 2, 2, 1, 1))), axis=4
    )  # dead can't progress

    # Testing: assumptions here should be verified. Are recovered or deceased individuals tested?
    # How is their data incorporated? Is it back-dated or listed at the test date?
    theta = np.concatenate(
        (np.zeros((1, 1, 1, 1, 1, 1)), theta), axis=0
    )  # susceptible don't test positive
    theta = np.concatenate(
        (theta, np.zeros((N_STATES, 1, 1, 1, 1, 1))), axis=3
    )  # recovered aren't tested
    theta = np.concatenate(
        (theta, np.zeros((N_STATES, 1, 1, 2, 1, 1))), axis=4
    )  # dead aren't tested

    # Recovery
    gamma = np.concatenate((np.zeros((1, N_AGES, 2, 1, 1, 1)), gamma), axis=0)
    gamma = np.concatenate(
        (gamma, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))), axis=4
    )  # dead can't recover

    # Lethality
    mu = np.concatenate(
        (np.zeros((N_INERT_STATES + N_HIDDEN_STATES, N_AGES, 2, 1, 1, 1)), mu), axis=0
    )
    mu = np.concatenate(
        (mu, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))), axis=3
    )  # recovered can't die

    # Generate a testval for y that follows our Euler-integrated ODE using the mean values for the parameters
    y = np.zeros((N_STATES, N_AGES, 2, 2, 2, N_T))
    y[0, :, 0, 0, 0, 0] = AGE_POP - e0
    y[1, :, 0, 0, 0, 0] = e0
    y0 = y[..., :1]
    for i in range(1, N_T):
        newly_exposed = (
            y0[:1] * np.sum(
                y0[N_INERT_STATES:, :, :, :1, :1] * beta[..., i - 1: i],
                axis=(0, 1, 2, 3, 4),
                keepdims=True,
            ) / POPULATION
        )

        disease_progressed = y0[1:-1] * sigma
        detections = y0[:, :, :1, :, :, :] * theta
        recoveries = y0[:, :, :, :1, :, :] * gamma
        deaths = y0[:, :, :, :, :1, :] * mu

        dy = np.concatenate((-newly_exposed, newly_exposed, disease_progressed), axis=0)
        z = np.zeros((1, N_AGES, 2, 2, 2, 1))
        dy += np.concatenate((z, -disease_progressed, z), axis=0)
        dy += np.concatenate((-detections, detections), axis=2)
        dy += np.concatenate((-recoveries, recoveries), axis=3)
        dy += np.concatenate((-deaths, deaths), axis=4)

        # += has side effects we don't want
        y0 = y0 + dy * DT

        y[..., i: i + 1] = y0

    return y + 0.01


y_kwargs = deterministic_ode(**{k: np.exp(v) for k, v in kwargs.items()})

SD = 2.0  # Default sigma for weakly informative Lognormal priors


# Add simple observation models for the data
def simple_obs_model(name, mu, c0=10.0, c1=0.04, normal=pm.Normal):
    mu = mu.sum(axis=(0, 1, 2, 3, 4))
    sd = np.sqrt(c0 ** 2 + c1 ** 2 * mu * mu)
    return normal(name, mu=mu, sd=sd, observed=lombardia[name])


def compare_y_to_data(y, normal=pm.Normal):
    y1 = y[..., I_FIRST:I_LAST + 1:N_PER_DAY]
    # Total confirmed cases: we assume this includes all detected cases.
    # By including recovered and deceased, we ensure this includes all past cases in addition to current ones.
    simple_obs_model('totale_casi', y1[1:, :, 1:, :, :], 2, 0.02, normal)
    simple_obs_model('totale_positivi', y1[1:, :, 1:, :1, :1], 2, 0.04, normal)
    # Deceased: just from detected cases
    simple_obs_model('deceduti', y1[:, :, 1:, :, 1:], 2, 0.01, normal)
    # Home isolation: includes only detected cases not admitted to the hospital
    # Should it include presymptomatic cases? Is it current or total? We treat it as total here.
    simple_obs_model('isolamento_domiciliare', y1[2:-2, :, 1:, :1, :1], 10, 0.10, normal)
    # Admitted with symptoms: corresponds to severe symptoms
    simple_obs_model('ricoverati_con_sintomi', y1[-2:-1, :, 1:, :1, :1], 2, 0.02, normal)
    # Intensive care: corresponds to critical symptoms
    simple_obs_model('terapia_intensiva', y1[-1:, :, 1:, :1, :1], 2, 0.02, normal)
    # Recovered: all detected, recovered cases that at one point required hospitalization
    # Is this "recovered" as translated on the GitHub page or "discharged, healed" as I would translate it?
    # If the former, this should also include other known cases once recovered.
    simple_obs_model('dimessi_guariti', y1[-2:, :, 1:, 1:, :1], 10, 0.10, normal)

    # Deaths by age
    f = y[:, :, 1:, :, 1:, idx_deaths].sum(axis=(0, 2, 3, 4))
    sd = np.sqrt(2.0 ** 2 + 0.02 ** 2 * f * f)
    normal(name='deaths_by_age', mu=f, sd=sd, observed=data_deaths)

    # Observe stats on breakdown by severity, age
    # Fraction of cases by severity
    f = y[N_INERT_STATES:, :, 1:, :1, :1, idx_severity].sum(axis=(1, 2, 3, 4))
    f = f / f.sum() * 100
    normal(name='percent_by_severity', mu=f, sd=4.0, observed=data_severity)
    # Fraction of cases by age
    f = y[N_INERT_STATES:, :, 1:, :, :, idx_cases].sum(axis=(0, 2, 3, 4))
    f = f / f.sum() * 100
    normal(name='percent_cases_by_age', mu=f, sd=2.0, observed=data_cases)
    # Lethality by age
    f = (
            y[:, :, 1:, :, 1:, idx_lethality].sum(axis=(0, 2, 3, 4))
            / y[:, :, 1:, :, :, idx_lethality].sum(axis=(0, 2, 3, 4))
            * 100
    )
    normal(name='lethality_by_age', mu=f, sd=4.0, observed=data_lethality)


def adjudicate_y(y, beta, sigma, theta, gamma, mu):
    # All living, non-recovered individuals in states above exposed can pass the virus
    # beta does not depend on age
    beta = beta[..., ERA_INDICES[:-1]]

    # All states except susceptible and critical can progress
    # Progression depends on age and detection status
    sigma = np.concatenate(
        [sigma, np.zeros((N_STATES - 2, N_AGES, 2, 1, 1, 1))], axis=3
    )  # recovered can't progress
    sigma = np.concatenate(
        [sigma, np.zeros((N_STATES - 2, N_AGES, 2, 2, 1, 1))], axis=4
    )  # dead can't progress

    # Testing: assumptions here should be verified. Are recovered or deceased individuals tested?
    # How is their data incorporated? Is it back-dated or listed at the test date?
    theta = np.concatenate([np.zeros((1, 1, 1, 1, 1, 1)), theta], axis=0)
    theta = np.concatenate(
        [theta, np.zeros((N_STATES, 1, 1, 1, 1, 1))], axis=3
    )  # recovered aren't tested
    theta = np.concatenate(
        [theta, np.zeros((N_STATES, 1, 1, 2, 1, 1))], axis=4
    )  # dead aren't tested

    # Recovery
    gamma = np.concatenate([np.zeros((1, N_AGES, 2, 1, 1, 1)), gamma], axis=0)
    gamma = np.concatenate(
        [gamma, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))], axis=4
    )  # dead can't recover

    # Lethality
    mu = np.concatenate(
        [np.zeros((N_INERT_STATES + N_HIDDEN_STATES, N_AGES, 2, 1, 1, 1)), mu], axis=0
    )
    mu = np.concatenate(
        [mu, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))], axis=3
    )  # recovered can't die

    # Observe to restrict initial conditions
    # Most categories should be zero at the start
    y0 = y[..., 0]
    pm.Normal(name='initial_detected', mu=y0[:, :, 1], sd=1, observed=0)
    pm.Normal(name='initial_recovered', mu=y0[:, :, 0, 1], sd=1, observed=0)
    pm.Normal(name='initial_deceased', mu=y0[:, :, 0, 0, 1], sd=1, observed=0)
    pm.Normal(
        name='initial_symptomatic',
        mu=y0[N_INERT_STATES + N_HIDDEN_STATES:, :, 0, 0, 0],
        sd=1,
        observed=0,
    )

    # Observe to enforce categories that should always be zero
    pm.Normal(name='recovered_dead', mu=y[:, :, :, 1, 1], sd=1, observed=0)
    pm.Normal(name='susceptible_recovered', mu=y[0, :, :, 1, :], sd=1, observed=0)
    pm.Normal(name='nonlethal_dead', mu=y[:3, :, :, :, 1], sd=1, observed=0)
    pm.Normal(name='confirmed_susceptible', mu=y[0, :, 1, :, :], sd=1, observed=0)

    # Compute the likelihood of each state based on the SDE and the prior state
    y0 = y[..., :-1]

    newly_exposed = (
            y0[:1]
            * (y0[N_INERT_STATES:, :, :, :1, :1] * beta).sum(axis=(0, 1, 2, 3, 4), keepdims=True)
            / POPULATION
    )
    disease_progressed = y0[1:-1] * sigma
    detections = y0[:, :, :1, :, :, :] * theta
    recoveries = y0[:, :, :, :1, :, :] * gamma
    deaths = y0[:, :, :, :, :1, :] * mu

    dy = np.concatenate([-newly_exposed, newly_exposed, disease_progressed], axis=0)
    z = np.zeros((1, N_AGES, 2, 2, 2, N_T - 1))
    dy += np.concatenate([z, -disease_progressed, z], axis=0)
    dy += np.concatenate([-detections, detections], axis=2)
    dy += np.concatenate([-recoveries, recoveries], axis=3)
    dy += np.concatenate([-deaths, deaths], axis=4)

    mu = y0 + DT * dy
    sd = np.sqrt(DT * ((2.0 ** 2) + (0.02 ** 2 * y0 * y0)))
    logp = pm.Normal.dist(mu=mu, sigma=sd).logp(y[..., 1:]).sum()
    pm.Potential(name='sde', var=logp)


def make_pymc_model():  # should be called within a pymc3.Model context

    # All living, non-recovered individuals in states above exposed can pass the virus
    # beta does not depend on age
    beta = pm.Lognormal(
        name='ibeta',
        mu=kwargs['beta'],
        sd=SD,
        testval=np.exp(kwargs['beta']),
        shape=(N_HIDDEN_STATES + N_LETHAL_STATES, 1, 2, 1, 1, N_ERAS),
    )

    # All states except susceptible and critical can progress
    # Progression depends on age and detection status
    sigma = pm.Lognormal(
        name='sigma',
        mu=kwargs['sigma'],
        sd=SD,
        testval=np.exp(kwargs['sigma']),
        shape=(N_STATES - 2, N_AGES, 2, 1, 1, 1),
    )

    # Testing: assumptions here should be verified. Are recovered or deceased individuals tested?
    # How is their data incorporated? Is it back-dated or listed at the test date?
    theta = pm.Lognormal(
        name='theta',
        mu=kwargs['theta'],
        sd=SD,
        testval=np.exp(kwargs['theta']),
        shape=(N_STATES - 1, 1, 1, 1, 1, 1),
    )

    # Recovery
    gamma = pm.Lognormal(
        name='gamma',
        mu=kwargs['gamma'],
        sd=SD,
        testval=np.exp(kwargs['gamma']),
        shape=(N_STATES - 1, N_AGES, 2, 1, 1, 1),
    )

    # Lethality
    mu = pm.Lognormal(
        name='mu',
        mu=kwargs['mu'],
        sd=SD,
        testval=np.exp(kwargs['mu']),
        shape=(N_LETHAL_STATES, N_AGES, 2, 1, 1, 1),
    )

    # All the states (enforce the total strictly)
    y = pm.Lognormal(
        name='cy',
        mu=9,
        sd=3,
        shape=(N_STATES, N_AGES, 2, 2, 2, N_T),
        testval=y_kwargs,
    )
    y /= y.sum(axis=(0, 2, 3, 4), keepdims=True)
    y *= np.reshape(AGE_POP, (1, N_AGES, 1, 1, 1, 1))
    y = pm.Deterministic(name='y', var=y)

    compare_y_to_data(y)
    adjudicate_y(y, beta, sigma, theta, gamma, mu)


class EarlyOut(Exception):
    """Exception to raise to stop nevergrad calculation early if logp is bad"""


class Logp:
    def __init__(self):
        self.value = 0.0

    def add_norm_logpdf(self, name, mu, sd, observed):
        offset = observed - mu
        self.value += np.sum(-(offset * offset) / (sd * sd) / 2 - np.log(sd))
        if not np.isfinite(self.value):
            raise EarlyOut


def func_nevergrad(
        e0,
        beta,
        sigma,
        theta,
        gamma,
        mu
):
    logp = Logp()
    y = deterministic_ode(
        np.exp(e0),
        np.exp(beta),
        np.exp(sigma),
        np.exp(theta),
        np.exp(gamma),
        np.exp(mu),
    )
    if np.any(y < 0):
        return float('inf')
    if np.any(y > POPULATION):
        return float('inf')
    try:
        compare_y_to_data(y, logp.add_norm_logpdf)
    except EarlyOut:
        return float('inf')
    return -logp.value


def run_nevergrad():
    # Guess initial values

    e0 = np.log(10) * np.ones(N_AGES)
    beta = - np.log(15) * np.ones((N_HIDDEN_STATES + N_LETHAL_STATES, 1, 2, 1, 1, N_ERAS))
    sigma = - np.log(3) * np.ones((N_STATES - 2, N_AGES, 2, 1, 1, 1))
    theta = - np.log(12) * np.ones((N_STATES - 1, 1, 1, 1, 1, 1))
    gamma = - np.log(120) * np.ones((N_STATES - 1, N_AGES, 2, 1, 1, 1))
    mu = - np.log(1e3) * np.ones((N_LETHAL_STATES, N_AGES, 2, 1, 1, 1))

    # OR, start from previous saved parameters

    # with open('seir/tmp.pkl', 'rb') as f:
    #     kwargs = pickle.load(f)
    # e0 = kwargs['e0']
    # beta = kwargs['beta']
    # sigma = kwargs['sigma']
    # theta = kwargs['theta']
    # gamma = kwargs['gamma']
    # mu = kwargs['mu']

    print(func_nevergrad(
        e0,
        beta,
        sigma,
        theta,
        gamma,
        mu
    ))

    # Use kwargs only
    instrum = ng.p.Instrumentation(
        e0=ng.p.Array(init=e0),
        beta=ng.p.Array(init=beta),
        sigma=ng.p.Array(init=sigma),
        theta=ng.p.Array(init=theta),
        gamma=ng.p.Array(init=gamma),
        mu=ng.p.Array(init=mu),
    )

    # optimizer = ng.optimizers.ParaPortfolio
    optimizer = ng.optimizers.TwoPointsDE
    kwargs, best_kwargs = util.fit_nevergrad_model(instrum, 2_000_000, optimizer, func_nevergrad,
                                                   num_workers=8, num_processes=8, save_after=2_000)
    with open('sample_sde.pkl', 'wb') as f:
        pickle.dump(kwargs, f)


def run_pymc3():
    with pm.Model():
        make_pymc_model()
        trace = pm.sample(
            200,
            tune=200,
            target_accept=0.99,
            compute_convergence_checks=False,
            chains=3,
            trace=pm.backends.Text(name='trace'),
        )


if __name__ == '__main__':
    run_nevergrad()
    # run_pymc3()
