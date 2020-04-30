import pickle

import nevergrad as ng
import numpy as np
import pandas as pd

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
totale_positivi = np.array(lombardia['totale_positivi'])
deceduti = np.array(lombardia['deceduti'])
isolamento_domiciliare = np.array(lombardia['isolamento_domiciliare'])
ricoverati_con_sintomi = np.array(lombardia['ricoverati_con_sintomi'])
terapia_intensiva = np.array(lombardia['terapia_intensiva'])
dimessi_guariti = np.array(lombardia['dimessi_guariti'])

# Susceptible, exposed, plus 5 levels of infectious, the first of which is asymptomatic
N_INERT_STATES = 2
N_HIDDEN_STATES = 1
N_LETHAL_STATES = 3  # any state with symptoms is considered lethal (even if lethality is very low)
N_STATES = N_INERT_STATES + N_HIDDEN_STATES + N_LETHAL_STATES
N_AGES = 3  # Number of age groups, <50, 50-70, >70
SD = 1  # Default sigma for weakly informative Lognormal priors

POPULATION = 10_060_574

AGE_DIST = np.array([0.5, 0.4, 0.1])
AGE_POP = POPULATION * AGE_DIST

N_PER_DAY = 4  # must divide 24 evenly
T = pd.date_range('8 Feb 2020', '1 May 2020', freq=f'{24//N_PER_DAY}h')
N_T = len(T)
T_RANGE = (T[-1] - T[0]).days
T0 = T[:-1]
T1 = T[1:]
# TODO: Both DT and SQRT_DT are all ones for now.
# If this resolution is good enough we should remove them.
# If not, but constant spacing is fine, we should make them scalars.
DT = 1 / N_PER_DAY  # (T1 - T0).days
# SQRT_DT = np.sqrt(DT)

ERA_STARTS = pd.to_datetime([
    '8 Mar 2020',
    '21 Mar 2020',
])
N_ERAS = len(ERA_STARTS) + 1  # Number of different time periods (characterized by different values of beta)
ERA_INDICES = np.sum(np.array(T) > ERA_STARTS[:, None], axis=0) if N_ERAS > 1 else [0] * N_T

I_FIRST = np.where(T == lombardia['data'].min())[0][0]
I_LAST = np.where(T == lombardia['data'].max())[0][0]

T_STATS = pd.to_datetime('16 April 2020')
I_STATS = np.where(T == T_STATS)[0][0]

T_STATS1 = pd.to_datetime('26 April 2020')
I_STATS1 = np.where(T == T_STATS)[0][0]


def norm_logpdf(x, mu, sd):
    """Return logpdf of normal distribution omitting constant terms"""
    offset = x - mu
    return -(offset * offset) / (sd * sd) / 2 - np.log(sd)


# Add simple observation models for the data
def simple_obs_model(x, mu, k=1.0):
    mu = mu.sum(axis=(0, 1, 2, 3, 4))
    sd = np.sqrt(1 ** 2 + 0.02 ** 2 * mu * mu) * k
    return norm_logpdf(x, mu, sd).sum()


def calc_mlogp(e0, beta, sigma, theta, gamma, mu, return_y=False):
# def calc_mlogp(t, e0, beta, sigma, theta, gamma, mu, return_y=False):

    #  state  ×  age  ×  confirmed  ×  recovered  ×  deceased  ×  time

    # t = T[0] + pd.to_timedelta(t, unit='d')

    e0 = np.exp(e0)
    beta = np.exp(beta)
    sigma = np.exp(sigma)
    theta = np.exp(theta)
    gamma = np.exp(gamma)
    mu = np.exp(mu)

    if np.any(e0 > AGE_POP):
        return float('inf')

    # All living, non-recovered individuals in states above exposed can pass the virus
    # beta does not depend on age
    beta = beta[..., ERA_INDICES[:-1]]

    # All states except susceptible and critical can progress
    # Progression depends on age and detection status
    sigma = np.concatenate((sigma, np.zeros((N_STATES - 2, N_AGES, 2, 1, 1, 1))), axis=3)  # recovered can't progress
    sigma = np.concatenate((sigma, np.zeros((N_STATES - 2, N_AGES, 2, 2, 1, 1))), axis=4)  # dead can't progress

    # Testing: assumptions here should be verified. Are recovered or deceased individuals tested?
    # How is their data incorporated? Is it back-dated or listed at the test date?
    theta = np.concatenate((np.zeros((1, 1, 1, 1, 1, 1)), theta), axis=0)  # susceptible don't test positive
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
    y[0, :, 0, 0, 0, 0] = AGE_POP - e0
    y[1, :, 0, 0, 0, 0] = e0
    y0 = y[..., :1]
    for i in range(1, N_T):
        # if T[i] >= t:  # could just freeze sigma by multiplying by a boolean along the time dimension
        if True:
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

            if np.any(y0 < 0) or np.any(y0 > POPULATION):
                return float('inf')

        y[..., i:i + 1] = y0

    # To make the log easier to work with, make sure no values are zero
    y += 0.01

    if return_y:
        return y

    logp = 0

    # # Compute the likelihood of each state based on the SDE and the prior state
    # y0 = y[..., :-1]
    #
    # newly_exposed = y0[:1] * np.sum(y0[N_INERT_STATES:, :, :, :1, :1] * beta,
    #                                 axis=(0, 1, 2, 3, 4), keepdims=True) / POPULATION
    # disease_progressed = y0[1:-1] * sigma
    # detections = y0[:, :, :1, :, :, :] * theta
    # recoveries = y0[:, :, :, :1, :, :] * gamma
    # deaths = y0[:, :, :, :, :1, :] * mu
    #
    # dy = np.concatenate((-newly_exposed, newly_exposed, disease_progressed), axis=0)
    # z = np.zeros((1, N_AGES, 2, 2, 2, N_T - 1))
    # dy += np.concatenate((z, -disease_progressed, z), axis=0)
    # dy += np.concatenate((-detections, detections), axis=2)
    # dy += np.concatenate((-recoveries, recoveries), axis=3)
    # dy += np.concatenate((-deaths, deaths), axis=4)
    #
    # mu = y0 + DT * dy
    # sd = np.sqrt(DT * ((10 ** 2) + (0.10 ** 2 * y0 * y0)))
    # logp += norm_logpdf(y[..., 1:], mu, sd).sum()  # pm.Normal.dist(mu=mu, sigma=sd).logp(y[..., 1:]).sum()

    y1 = y[..., I_FIRST:I_LAST + 1:N_PER_DAY]
    # Total confirmed cases: we assume this includes all detected cases.
    # By including recovered and deceased, we ensure this includes all past cases in addition to current ones.
    logp += simple_obs_model(totale_casi, y1[1:, :, 1:, :, :], k=1/2)
    # Just the current cases:
    logp += simple_obs_model(totale_positivi, y1[1:, :, 1:, :1, :1])
    # Deceased: just from detected cases
    logp += simple_obs_model(deceduti, y1[:, :, 1:, :, 1:], k=1/4)
    # Home isolation: includes only active detected cases not admitted to the hospital
    logp += simple_obs_model(isolamento_domiciliare, y1[2:-2, :, 1:, :1, :1])
    # Admitted with symptoms (current): corresponds to severe symptoms
    logp += simple_obs_model(ricoverati_con_sintomi, y1[-2:-1, :, 1:, :1, :1])
    # Intensive care (current): corresponds to critical symptoms
    logp += simple_obs_model(terapia_intensiva, y1[-1:, :, 1:, :1, :1])
    # Recovered: all detected, recovered cases that at one point required hospitalization
    # Is this "recovered" as translated on the GitHub page or "discharged, healed" as I would translate it?
    # If the former, this should also include other known cases once recovered.
    logp += simple_obs_model(dimessi_guariti, y1[-2:, :, 1:, 1:, :1])

    # # Observe to restrict initial conditions
    # # Most categories should be zero at the start
    # y0 = y[..., 0]
    # logp += norm_logpdf(0, y0[:, :, 1], 10).sum()
    # logp += norm_logpdf(0, y0[:, :, 0, 1], 10).sum()
    # logp += norm_logpdf(0, y0[:, :, 0, 0, 1], 10).sum()
    # logp += norm_logpdf(0, y0[N_INERT_STATES + N_HIDDEN_STATES:, :, 0, 0, 0], 10).sum()
    # # Assume initial exposed and asymptomatic infectious numbers are small with a wide prior
    # logp += norm_logpdf(500, y0[1:N_INERT_STATES + N_HIDDEN_STATES, :, 0, 0, 0], 1_000).sum()

    # # Observe to enforce categories that should always be zero
    # logp += norm_logpdf(0, y[:, :, :, 1, 1], 10).sum()
    # logp += norm_logpdf(0, y[0, :, :, 1, :], 10).sum()
    # logp += norm_logpdf(0, y[:3, :, :, :, 1], 10).sum()
    # logp += norm_logpdf(0, y[0, :, 1, :, :], 10).sum()

    # Observe to enforce assumption of high testing for hospitalized cases (few undetected cases in hospital)
    logp += norm_logpdf(0, y[-2:, :, 0, :, :], 1).sum()

    # Observe to enforce total population
    total = y.sum(axis=(0, 1, 2, 3, 4))
    logp += norm_logpdf(POPULATION, total, 10).sum()

    # Observe statistics from April 16
    y2 = y[..., I_STATS]
    # Fraction of cases by severity
    f = y2[N_INERT_STATES:, :, 1:, :, :].sum(axis=(1, 2, 3, 4))
    f = f / f.sum() * (100 - 15.2)  # scale by specified symptom levels
    # Combine "pauci-sintomatico" and "lieve"
    logp += norm_logpdf(np.array([11.9, 16.9 + 35.6, 18.2, 2.2]), f, 1.0).sum()
    # Fraction of cases by age
    f = y2[N_INERT_STATES:, :, 1:, :, :].sum(axis=(0, 2, 3, 4))
    f = f / f.sum() * 100
    logp += norm_logpdf(np.array([1.7 + 27, 33.5, 37.8]), f, 0.1).sum()
    # Fraction deaths by age
    f = y2[:, :, 1:, :, 1:].sum(axis=(0, 2, 3, 4))
    f = f / f.sum() * 100
    logp += norm_logpdf(np.array([0 + 0 + 0 + 0.2 + 0.9, 3.8 + 11.3, 30.8 + 40.5 + 12.4]), f, 0.1).sum()
    # Lethality by age
    f = y2[:, :, 1:, :, 1:].sum(axis=(0, 2, 3, 4)) / y2[:, :, 1:, :, :].sum(axis=(0, 2, 3, 4)) * 100
    deaths = np.array([2, 0, 7, 41, 180, 776, 2330, 6333, 8312, 2549])
    lethality = np.array([0.2, 0, 0.1, 0.3, 0.9,  2.6, 9.6, 24.3, 30.5, 25.1])
    cases = deaths / lethality
    cases[np.isnan(cases)] = 0
    combined_lethality = np.array([
        deaths[:5].sum() / cases[:5].sum(),
        deaths[5:7].sum() / cases[5:7].sum(),
        deaths[7:].sum() / cases[7:].sum(),
    ])
    logp += norm_logpdf(combined_lethality, f, 0.1).sum()

    # Observe to weakly enforce plausible fraction of undetected cases on April 16
    percent_undetected = 100 * y2[1:, :, :1].sum() / y2[1:, :, :].sum()
    # counter-intuitively, sd of 1 is weak because of quantity of other observations
    logp += norm_logpdf(percent_undetected, 50, 1)

    # Observe statistics from April 26
    y2 = y[..., I_STATS1]
    # Fraction of cases by severity
    f = y2[N_INERT_STATES:, :, 1:, :, :].sum(axis=(1, 2, 3, 4))
    f = f / f.sum() * (100 - 15.4)  # scale by specified symptom levels
    # Combine "pauci-sintomatico" and "lieve"
    logp += norm_logpdf(np.array([13.3, 16.3 + 35.5, 17.5, 2.0]), f, 1.0).sum()
    # Fraction of cases by age
    f = y2[N_INERT_STATES:, :, 1:, :, :].sum(axis=(0, 2, 3, 4))
    f = f / f.sum() * 100
    logp += norm_logpdf(np.array([1.8 + 27.4, 31.8, 39.0]), f, 0.1).sum()
    # Fraction deaths by age
    f = y2[:, :, 1:, :, 1:].sum(axis=(0, 2, 3, 4))
    f = f / f.sum() * 100
    logp += norm_logpdf(np.array([0 + 0 + 0 + 0.2 + 0.9, 3.6 + 10.9, 29 + 40.6 + 14.7]), f, 0.1).sum()
    # Lethality by age
    f = y2[:, :, 1:, :, 1:].sum(axis=(0, 2, 3, 4)) / y2[:, :, 1:, :, :].sum(axis=(0, 2, 3, 4)) * 100
    deaths = np.array([2, 0, 8, 49, 223, 903, 2708, 7191, 10050, 3646])
    lethality = np.array([0.1, 0, 0.1, 0.3, 0.9,  2.5, 9.8, 24.1, 28.9, 24.6])
    cases = deaths / lethality
    cases[np.isnan(cases)] = 0
    combined_lethality = np.array([
        deaths[:5].sum() / cases[:5].sum(),
        deaths[5:7].sum() / cases[5:7].sum(),
        deaths[7:].sum() / cases[7:].sum(),
    ])
    logp += norm_logpdf(combined_lethality, f, 0.1).sum()

    # Observe to weakly enforce plausible fraction of undetected cases on April 26
    percent_undetected = 100 * y2[1:, :, :1].sum() / y2[1:, :, :].sum()
    logp += norm_logpdf(percent_undetected, 50, 1)

    return -logp


def main():

    # Number per age group initially exposed
    e0 = np.ones(N_AGES) * 10
    e0 = np.log(e0)

    t = 1.0

    beta = - np.log(15) * np.ones((N_HIDDEN_STATES + N_LETHAL_STATES, 1, 2, 1, 1, N_ERAS))
    sigma = - np.log(3) * np.ones((N_STATES - 2, N_AGES, 2, 1, 1, 1))
    theta = - np.log(12) * np.ones((N_STATES - 1, 1, 1, 1, 1, 1))
    gamma = - np.log(120) * np.ones((N_STATES - 1, N_AGES, 2, 1, 1, 1))
    mu = - np.log(1e3) * np.ones((N_LETHAL_STATES, N_AGES, 2, 1, 1, 1))

    print(calc_mlogp(
        # t,
        e0,
        beta,
        sigma,
        theta,
        gamma,
        mu
    ))

    # Use kwargs only
    instrum = ng.p.Instrumentation(
        # t=ng.p.Scalar(init=t, lower=0, upper=(lombardia['data'].min() - T[0]).days - 4),
        e0=ng.p.Array(init=e0),
        beta=ng.p.Array(init=beta),
        sigma=ng.p.Array(init=sigma),
        theta=ng.p.Array(init=theta),
        gamma=ng.p.Array(init=gamma),
        mu=ng.p.Array(init=mu),
    )

    # optimizer = ng.optimizers.ParaPortfolio
    # optimizer = ng.optimizers.OnePlusOne
    # optimizer = ng.optimizers.CMandAS3
    optimizer = ng.optimizers.TwoPointsDE  # This one seems to be the best
    # optimizer = ng.optimizers.CMA
    # optimizer = ng.optimizers.SQPCMA
    # optimizer = ng.optimizers.TripleCMA
    # optimizer = ng.optimizers.PSO
    # optimizer = ng.optimizers.ScrHammersleySearch
    # optimizer = ng.optimizers.ASCMA2PDEthird
    # optimizer = ng.optimizers.ASCMADEQRthird
    # optimizer = ng.optimizers.ASCMADEthird
    # optimizer = ng.optimizers.CMandAS3
    # optimizer = ng.optimizers.DifferentialEvolution(
    #     initialization='gaussian',  # LHS, QR, gaussian
    #     scale=1.0,
    #     recommendation='optimistic',
    #     crossover='twopoints',  # dimension, random, onepoint, twopoints, parametrization
    #     F1=0.8,
    #     F2=0.8,
    #     popsize='large',  # [int], standard, dimension, large
    # )
    # optimizer = ng.optimizers.ParametrizedCMA(
    #     scale=0.2,
    #     popsize=16,  # [int], standard, dimension, large
    #     diagonal=False,
    #     fcmaes=False,
    # )
    kwargs, best_kwargs = util.fit_nevergrad_model(instrum, 2_000_000, optimizer, calc_mlogp,
                                                   num_workers=8, save_after=5_000)
    with open('newsde.pkl', 'wb') as f:
        pickle.dump(kwargs, f)
    with open('newsde_best.pkl', 'wb') as f:
        pickle.dump(best_kwargs, f)


if __name__ == '__main__':
    main()
