import pickle

import nevergrad as ng
import numpy as np
import pandas as pd

from seir import util

print('Reloading!')

data = pd.read_csv(
    'https://github.com/pcm-dpc/COVID-19/raw/master/dati-regioni/dpc-covid19-ita-regioni.csv',
    parse_dates=['data'],
)

# The data for some regions might be causing problems. There are few cases here and the
# number recovered lags the deaths in the early days that suggests reporting dynamics not captured by this model.
# That could also be overcome by increasing the low value uncertainty but with as few cases as they have it would
# probably wash out any effect these regions have on the model.
ds = (
    data
    .set_index(['denominazione_regione', 'data'])
    .drop(columns=['lat', 'long', 'codice_regione', 'stato', 'note_it', 'note_en', 'casi_testati'])
    .to_xarray()
).drop_sel(denominazione_regione=['Basilicata', 'Marche', 'Molise', "Valle d'Aosta"])
region_names = list(ds['denominazione_regione'].values)

t = ds['data'].values
n_time = len(t)
n_regions = len(ds['denominazione_regione'])

era_starts = np.array(pd.to_datetime([
    # '1 Mar 2020',
    '8 Mar 2020',
    '21 Mar 2020',
    '14 Apr 2020',
    '4 May 2020',
    # '3 Jun 2020',
]))
n_eras = len(era_starts) + 1
era_indices = np.sum(np.array(t) > era_starts[:, None], axis=0) if n_eras > 1 else [0] * n_time

# Calculate an estimate for new tests by smoothing total tests and taking the difference
# Add a bias to avoid any zero test days
# This should have very little effect since most days have at least tens of tests
# Though it could force undetected cases toward zero just because of lack of testing
# but that should be a weak effect
# The detection rate will be modeled as proportional to the square root of the new test count
new_tests = ds['tamponi'].rolling({'data': 7}, center=False, min_periods=1).mean()
new_tests = new_tests.diff(dim='data') + 0.1
sqrt_new_tests = np.sqrt(new_tests).values

min_threshold = -100
max_threshold = 1e7


def ode(initial_exposed, initial_infectious, beta, beta_detected_ratio, sigma, theta,
        gamma_mu, gamma_detected, mu_detected, mu_ratio):
    beta = beta[..., era_indices]
    beta_detected_ratio = beta_detected_ratio[..., era_indices]
    # Limit possible % detections to avoid negative infectious cases
    # If % detected in a single day is ever actually this high we should make a better model for theta
    theta = np.minimum(0.95, theta * 1e-2 * sqrt_new_tests)

    mu_ratio = np.concatenate([[1], mu_ratio])[era_indices]

    exposed = np.inf * np.ones((n_regions, n_time))
    infectious = np.inf * np.ones((n_regions, n_time))
    detected = np.inf * np.ones((n_regions, n_time))
    recovered = np.inf * np.ones((n_regions, n_time))
    dead = np.inf * np.ones((n_regions, n_time))

    exposed[:, 0] = initial_exposed
    infectious[:, 0] = initial_infectious
    detected[:, 0] = ds['totale_casi'].values[:, 0]
    recovered[:, 0] = ds['dimessi_guariti'].values[:, 0]
    dead[:, 0] = ds['deceduti'].values[:, 0]

    for i in range(1, n_time):
        newly_exposed = (
                beta[..., i - 1] * infectious[..., i - 1] +
                beta[..., i - 1] * beta_detected_ratio[..., i - 1] * detected[..., i - 1]
        )
        progressed = sigma * exposed[..., i - 1]
        detections = theta[..., i - 1] * infectious[..., i - 1]
        recoveries_detected = gamma_detected * detected[..., i - 1]
        # deaths_detected = mu_detected * detected[..., i - 1]
        deaths_detected = mu_detected * mu_ratio[i - 1] * detected[..., i - 1]

        exposed_d = newly_exposed - progressed
        infectious_d = progressed - detections - gamma_mu * infectious[..., i - 1]
        detected_d = detections - recoveries_detected - deaths_detected
        recovered_d = recoveries_detected
        dead_d = deaths_detected

        exposed[..., i] = exposed[..., i - 1] + exposed_d
        infectious[..., i] = infectious[..., i - 1] + infectious_d
        detected[..., i] = detected[..., i - 1] + detected_d
        recovered[..., i] = recovered[..., i - 1] + recovered_d
        dead[..., i] = dead[..., i - 1] + dead_d

        if (
                np.any(exposed[..., i] < min_threshold) or
                np.any(infectious[..., i] < min_threshold) or
                np.any(detected[..., i] < min_threshold) or
                np.any(recovered[..., i] < min_threshold) or
                np.any(dead[..., i] < min_threshold) or
                np.any(exposed[..., i] > max_threshold) or
                np.any(infectious[..., i] > max_threshold) or
                np.any(detected[..., i] > max_threshold) or
                np.any(recovered[..., i] > max_threshold) or
                np.any(dead[..., i] > max_threshold)
        ):
            break  # score will be inf

        exposed[..., i] = np.maximum(0, exposed[..., i])
        infectious[..., i] = np.maximum(0, infectious[..., i])
        detected[..., i] = np.maximum(0, detected[..., i])
        recovered[..., i] = np.maximum(0, recovered[..., i])
        dead[..., i] = np.maximum(0, dead[..., i])

    return exposed, infectious, detected, recovered, dead


def calc_score(exposed, infectious, detected, recovered, dead):
    score = 0

    y = ds['totale_positivi'].values
    var = 10 ** 2 + y * y * (0.1 ** 2)
    x = y - detected
    x = x * x / var
    score += x.sum()

    y = ds['deceduti'].values
    var = 5 ** 2 + y * y * (0.1 ** 2)
    x = y - dead
    x = x * x / var
    score += x.sum()

    y = ds['dimessi_guariti'].values
    var = 40 ** 2 + y * y * (0.2 ** 2)
    x = y - recovered
    x = x * x / var
    score += x.sum()

    return score


def transform(initial_exposed, initial_infectious, beta, beta_detected_ratio, sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio):
    initial_exposed = np.exp(initial_exposed)
    initial_infectious = np.exp(initial_infectious)
    beta = np.exp(beta)
    beta_detected_ratio = (np.tanh(beta_detected_ratio) + 1) / 2
    theta = np.exp(theta)
    mu_detected = np.exp(mu_detected)
    mu_ratio = (np.tanh(mu_ratio) + 3) / 4  # in (0.5, 1)
    return initial_exposed, initial_infectious, beta, beta_detected_ratio, sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio


def func(initial_exposed, initial_infectious, beta, beta_detected_ratio, sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio):
    return calc_score(
        *ode(
            *transform(
                initial_exposed, initial_infectious, beta, beta_detected_ratio, sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio
            )
        )
    )


def main():

    beta = 0.05 * np.ones((n_regions, n_eras))
    beta_detected_ratio = 0.5 * np.ones((n_regions, n_eras))
    sigma = 0.2
    theta = 0.1 * np.ones((n_regions, 1))
    gamma_mu = 0.1
    gamma_detected = 0.1
    mu_detected = 0.1 * np.ones(n_regions)
    mu_ratio = np.ones(n_eras-1)

    initial_exposed = 200 * np.ones(n_regions)
    initial_infectious = 20 * np.ones(n_regions)

    print('Initial guess:', calc_score(*ode(initial_exposed, initial_infectious, beta,
                                            beta_detected_ratio, sigma, theta, gamma_mu,
                                            gamma_detected, mu_detected, mu_ratio)))

    instrum = ng.p.Instrumentation(
        initial_exposed=ng.p.Array(init=np.log(initial_exposed)),
        initial_infectious=ng.p.Array(init=np.log(initial_infectious)),
        beta=ng.p.Array(init=np.log(beta)),
        beta_detected_ratio=ng.p.Array(init=beta_detected_ratio*0),
        sigma=ng.p.Log(init=sigma),
        theta=ng.p.Array(init=np.log(theta)),
        gamma_mu=ng.p.Log(init=gamma_mu),
        gamma_detected=ng.p.Log(init=gamma_detected),
        mu_detected=ng.p.Array(init=mu_detected),
        mu_ratio=ng.p.Array(init=mu_ratio * 5),
    )
    print('Initial instrumentation:', func(**instrum.kwargs))

    # Or use last solution as starting point
    with open('tmp_best21.pkl', 'rb') as f:
        kwargs = pickle.load(f)
    beta = kwargs['beta']
    beta_detected_ratio = kwargs['beta_detected_ratio']
    sigma = kwargs['sigma']
    theta = kwargs['theta']
    gamma_mu = kwargs['gamma_mu']
    gamma_detected = kwargs['gamma_detected']
    mu_detected = kwargs['mu_detected']
    mu_ratio = kwargs['mu_ratio']
    initial_exposed = kwargs['initial_exposed']
    initial_infectious = kwargs['initial_infectious']

    print('Last solution:', func(initial_exposed, initial_infectious, beta, beta_detected_ratio,
                                 sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio))

    # sigma /= 1.2
    # gamma_mu /= 1.2

    # beta = np.delete(beta, [-1], axis=1)
    # beta_detected_ratio = np.delete(beta_detected_ratio, [-1], axis=1)
    # mu_ratio = np.delete(mu_ratio, [-1], axis=0)
    #
    # # Replace bad regions stuck in weird space
    # for replace_me, replace_with in [
    #     # ('P.A. Trento', 'Liguria'),
    #     # ('P.A. Bolzano', 'Liguria'),
    #     # ('Abruzzo', 'Friuli Venezia Giulia'),
    #     # ('Campania', 'Calabria'),
    #     # ('Sicilia', 'Veneto'),
    #     # ('Lombardia', 'Piemonte'),
    #     # ('Piemonte', 'Puglia'),
    #     # ('Umbria', 'Veneto'),
    # ]:
    #     replace_me = region_names.index(replace_me)
    #     replace_with = region_names.index(replace_with)
    #     beta[replace_me] = beta[replace_with]
    #     beta_detected_ratio[replace_me] = beta_detected_ratio[replace_with]
    #     theta[replace_me] = theta[replace_with]
    #     mu_detected[replace_me] = mu_detected[replace_with]

    # print('Perturbed solution:', func(initial_exposed, initial_infectious, beta, beta_detected_ratio,
    #                                   sigma, theta, gamma_mu, gamma_detected, mu_detected, mu_ratio))

    instrum = ng.p.Instrumentation(
        initial_exposed=ng.p.Array(init=initial_exposed),
        initial_infectious=ng.p.Array(init=initial_infectious),
        beta=ng.p.Array(init=beta),
        beta_detected_ratio=ng.p.Array(init=beta_detected_ratio),
        sigma=ng.p.Log(init=sigma),
        theta=ng.p.Array(init=theta),
        gamma_mu=ng.p.Log(init=gamma_mu),
        gamma_detected=ng.p.Log(init=gamma_detected),
        mu_detected=ng.p.Array(init=mu_detected),
        mu_ratio=ng.p.Array(init=mu_ratio),
    )
    print('Initial instrumentation:', func(**instrum.kwargs))

    # recommendation = ng.optimizers.TwoPointsDE(instrum, budget=1_000).minimize(func)
    # print('Early recommendation:', func(**recommendation.kwargs))

    kwargs, best_kwargs = util.fit_nevergrad_model(
        instrum, 1_000_000, ng.optimizers.MultiCMA, func, num_workers=9, num_processes=9, save_after=1_000
    )
    print('Recommendation:', func(**kwargs))
    print('Best scorer:', func(**best_kwargs))


if __name__ == '__main__':
    main()
