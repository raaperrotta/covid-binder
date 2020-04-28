import pickle

import nevergrad as ng
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import util

np.seterr(invalid='ignore', over='ignore')

regioni = pd.read_csv(
    'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data']
)
lombardia = regioni[regioni['denominazione_regione'] == 'Lombardia'].copy()
lombardia['data'] = pd.to_datetime(lombardia['data'].dt.date)  # Drop the time
lombardia.sort_values('data', inplace=True)

totale_casi = np.array(lombardia['totale_casi'])
deceduti = np.array(lombardia['deceduti'])
isolamento_domiciliare = np.array(lombardia['isolamento_domiciliare'])
ricoverati_con_sintomi = np.array(lombardia['ricoverati_con_sintomi'])
terapia_intensiva = np.array(lombardia['terapia_intensiva'])
dimessi_guariti = np.array(lombardia['dimessi_guariti'])

# Susceptible, exposed, plus 5 levels of infectious, the first of which is asymptomatic
N_INERT_STATES = 2
N_HIDDEN_STATES = 1
N_LETHAL_STATES = 4  # any state with symptoms is considered lethal (even if lethality is very low)
N_STATES = N_INERT_STATES + N_HIDDEN_STATES + N_LETHAL_STATES
N_AGES = 1  # Number of age groups
SD = 1  # Default sigma for weakly informative Lognormal priors

POPULATION = 10_060_574

T = pd.date_range('20 Feb 2020', '1 May 2020', freq='1d')
N_T = len(T)
T0 = T[:-1]
T1 = T[1:]
# TODO: Both DT and SQRT_DT are all ones for now.
# If this resolution is good enough we should remove them.
# If not, but constant spacing is fine, we should make them scalars.
DT = 1.0  # (T1 - T0).days
SQRT_DT = np.sqrt(DT)

ERA_STARTS = pd.to_datetime([
    '1 Jan 1900',  # before start of T for easier indexing
    '8 Mar 2020',
    '21 Mar 2020',
])
N_ERAS = len(ERA_STARTS)  # Number of different time periods (characterized by different values of beta)
ERA_INDICES = interp1d(ERA_STARTS.astype(int), list(range(N_ERAS)), kind='previous', fill_value='extrapolate')(
    T.astype(int)).astype(int)

T_STATS = pd.to_datetime('16 April 2020')
I_STATS = np.where(T == T_STATS)[0][0]
I_FIRST = np.where(T == lombardia['data'].min())[0][0]
I_LAST = np.where(T == lombardia['data'].max())[0][0]


def norm_logpdf(x, mu, sd):
    """Return logpdf of normal distribution omitting constant terms"""
    offset = x - mu
    return -(offset * offset) / (sd * sd) / 2 - np.log(sd)


# Add simple observation models for the data
def simple_obs_model(x, mu):
    mu = mu.sum(axis=(0, 1, 2, 3, 4))
    sd = np.sqrt(10 ** 2 + 0.10 ** 2 * mu * mu)
    return norm_logpdf(x, mu, sd).sum()


def calc_mlogp(e0, beta, sigma, theta, gamma, mu, return_y=False):
    #  state  ×  age  ×  confirmed  ×  recovered  ×  deceased  ×  time

    e0 = np.exp(e0)
    beta = np.exp(beta)
    sigma = np.exp(sigma)
    theta = np.exp(theta)
    gamma = np.exp(gamma)
    mu = np.exp(mu)

    # All living, non-recovered individuals in states above exposed can pass the virus
    # beta does not depend on age
    beta = beta[..., ERA_INDICES[:-1]]

    # All states except susceptible and critical can progress
    # Progression depends on age and detection status
    sigma = np.concatenate((sigma, np.zeros((N_STATES - 2, N_AGES, 2, 1, 1, 1))), axis=3)  # recovered can't progress
    sigma = np.concatenate((sigma, np.zeros((N_STATES - 2, N_AGES, 2, 2, 1, 1))), axis=4)  # dead can't progress

    # Testing: assumptions here should be verified. Are recovered or deceased individuals tested?
    # How is their data incorporated? Is it back-dated or listed at the test date?
    theta = np.concatenate((np.zeros((N_INERT_STATES + N_HIDDEN_STATES, 1, 1, 1, 1, 1)), theta), axis=0)
    theta = np.concatenate((theta, np.zeros((N_STATES, 1, 1, 1, 1, 1))), axis=3)  # recovered aren't tested
    theta = np.concatenate((theta, np.zeros((N_STATES, 1, 1, 2, 1, 1))), axis=4)  # dead aren't tested

    # Recovery
    gamma = np.concatenate((np.zeros((1, N_AGES, 2, 1, 1, 1)), gamma), axis=0)
    gamma = np.concatenate((gamma, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))), axis=4)  # dead can't recover

    # Lethality
    mu = np.concatenate((np.zeros((N_INERT_STATES + N_HIDDEN_STATES, N_AGES, 2, 1, 1, 1)), mu), axis=0)
    mu = np.concatenate((mu, np.zeros((N_STATES, N_AGES, 2, 1, 1, 1))), axis=3)  # recovered can't die

    # Generate a testval for y that follows our Euler-integrated ODE using the mean values for the parameters
    y = np.zeros((N_STATES, N_AGES, 2, 2, 2, N_T))
    y[..., 0] = e0
    y0 = y[..., :1]
    for i in range(1, N_T):
        newly_exposed = y0[:1] * np.sum(y0[N_INERT_STATES:, :, :, :1, :1] * beta[..., i - 1:i],
                                        axis=(0, 1, 2, 3, 4), keepdims=True) / POPULATION
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

        y[..., i:i + 1] = y0

    # To make the log easier to work with, make sure no values are zero
    y += 0.01

    if return_y:
        return y

    # Compute the likelihood of each state based on the SDE and the prior state
    y0 = y[..., :-1]

    newly_exposed = y0[:1] * np.sum(y0[N_INERT_STATES:, :, :, :1, :1] * beta,
                                    axis=(0, 1, 2, 3, 4), keepdims=True) / POPULATION
    disease_progressed = y0[1:-1] * sigma
    detections = y0[:, :, :1, :, :, :] * theta
    recoveries = y0[:, :, :, :1, :, :] * gamma
    deaths = y0[:, :, :, :, :1, :] * mu

    dy = np.concatenate((-newly_exposed, newly_exposed, disease_progressed), axis=0)
    z = np.zeros((1, N_AGES, 2, 2, 2, N_T - 1))
    dy += np.concatenate((z, -disease_progressed, z), axis=0)
    dy += np.concatenate((-detections, detections), axis=2)
    dy += np.concatenate((-recoveries, recoveries), axis=3)
    dy += np.concatenate((-deaths, deaths), axis=4)

    mu = y0 + DT * dy
    sd = np.sqrt(DT * ((10 ** 2) + (0.10 ** 2 * y0 * y0)))
    logp = norm_logpdf(y[..., 1:], mu, sd).sum()  # pm.Normal.dist(mu=mu, sigma=sd).logp(y[..., 1:]).sum()

    y1 = y[..., I_FIRST:I_LAST + 1]
    # Total confirmed cases: we assume this includes all detected cases.
    # By including recovered and deceased, we ensure this includes all past cases in addition to current ones.
    logp += simple_obs_model(totale_casi, y1[:, :, 1:, :, :])
    # Deceased: just from detected cases
    logp += simple_obs_model(deceduti, y1[:, :, 1:, :, 1:])
    # Home isolation: includes only detected cases not admitted to the hospital
    # Should it include presymptomatic cases? Is it current or total? We treat it as total here.
    logp += simple_obs_model(isolamento_domiciliare, y1[:-2, :, 1:, :, :])
    # Admitted with symptoms: corresponds to severe symptoms
    logp += simple_obs_model(ricoverati_con_sintomi, y1[-2:-1, :, 1:, :, :])
    # Intensive care: corresponds to critical symptoms
    logp += simple_obs_model(terapia_intensiva, y1[-1:, :, 1:, :, :])
    # Recovered: all detected, recovered cases that at one point required hospitalization
    # Is this "recovered" as translated on the GitHub page or "discharged, healed" as I would translate it?
    # If the former, this should also include other known cases once recovered.
    logp += simple_obs_model(dimessi_guariti, y1[-2:, :, 1:, 1:, :1])

    # Observe to restrict initial conditions
    # Most categories should be zero at the start
    y0 = y[..., 0]
    logp += norm_logpdf(0, y0[:, :, 1], 10).sum()
    logp += norm_logpdf(0, y0[:, :, 0, 1], 10).sum()
    logp += norm_logpdf(0, y0[:, :, 0, 0, 1], 10).sum()
    logp += norm_logpdf(0, y0[N_INERT_STATES + N_HIDDEN_STATES:, :, 0, 0, 0], 10).sum()
    # Assume initial exposed and asymptomatic infectious numbers are small with a wide prior
    logp += norm_logpdf(500, y0[1:N_INERT_STATES + N_HIDDEN_STATES, :, 0, 0, 0], 1_000).sum()

    # Observe to enforce categories that should always be zero
    logp += norm_logpdf(0, y[:, :, :, 1, 1], 10).sum()
    logp += norm_logpdf(0, y[0, :, :, 1, :], 10).sum()
    logp += norm_logpdf(0, y[:3, :, :, :, 1], 10).sum()
    logp += norm_logpdf(0, y[0, :, 1, :, :], 10).sum()

    # Observe to enforce total population
    total = y.sum(axis=(0, 1, 2, 3, 4))
    logp += norm_logpdf(POPULATION, total, 1).sum()

    return -logp


def main():

    e0 = np.ones((N_STATES, N_AGES, 2, 2, 2)) * 10
    e0[1, 0, 0, 0, 0] = 1_000
    e0[0, 0, 0, 0, 0] = 0
    e0[0, 0, 0, 0, 0] = POPULATION - e0.sum()
    e0 = np.log(e0 + 0.001)

    beta = - np.log(100) * np.ones((N_HIDDEN_STATES + N_LETHAL_STATES, 1, 2, 1, 1, N_ERAS))
    sigma = - np.log(20) * np.ones((N_STATES - 2, N_AGES, 2, 1, 1, 1))
    theta = - np.log(100) * np.ones((N_LETHAL_STATES, 1, 1, 1, 1, 1))
    gamma = - np.log(20) * np.ones((N_STATES - 1, N_AGES, 2, 1, 1, 1))
    mu = - np.log(100) * np.ones((N_LETHAL_STATES, N_AGES, 2, 1, 1, 1))

    print(calc_mlogp(e0, beta, sigma, theta, gamma, mu))

    instrum = ng.p.Instrumentation(
        e0=ng.p.Array(init=e0),
        beta=ng.p.Array(init=beta),
        sigma=ng.p.Array(init=sigma),
        theta=ng.p.Array(init=theta),
        gamma=ng.p.Array(init=gamma),
        mu=ng.p.Array(init=mu),
    )

    kwargs, best_kwargs = util.fit_nevergrad_model(instrum, 2_000_000, ng.optimizers.ParaPortfolio, calc_mlogp,
                                                   num_workers=1_000)
    with open('newsde.pkl', 'wb') as f:
        pickle.dump(kwargs, f)
    with open('newsde_best.pkl', 'wb') as f:
        pickle.dump(best_kwargs, f)


if __name__ == '__main__':
    main()
