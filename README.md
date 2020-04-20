# Modeling COVID-19

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raaperrotta/covid-binder/master)


## 17 April 2020:
I'm playing catch-up with notes here.
As of today I have a SEIR model (simplified derivative of the SEIRS+ library) that I can fit against data from Lombardy using nevergrad and pymc3. Nevergrad can achieve a good fit of the data but gives no characterization of the uncertainty. If I use the nevergrad solution to create priors for pymc3 I get good-looking, consistent results with sensible uncertainty expressed.

While I was able to get nevergrad to infer the start times for the virus spread and for countermeasures, the results were very noisy. Since we know when the countermeasures were put in place, I fixed those times and I chose to fix the start date and allow both the initial exposed and initial infectious (with mild symptoms) counts to be fit.

I ran the model using Lognormal priors with sd=0.55 and 1.
The more confident priors led to better looking results (of course) but it is hard to be confident that covers all plausible explanations.

One of my current limitations is the runtime of my pymc model. I can use the pymc3.ode.DifferentialEquation class to augment my ODE to compute the gradient of the solution (super-cool) which enables gradient-based samplers on this ODE problem. But it is slow! And more cripplingly, it slows down significantly as it runs. Initial computations take about a second apiece but after tens of samples each one takes a minute or more. They quickly slow down beyond the limits of my patience.

Given the slowdown, I thought the culprit might be the solution caching in the DifferentialEquation class so I created my own version of it and removed the caching. That should cause the ODE to be solved twice, once for the state and once for the gradient, but might solve the slow-down; however, it didn't.

I suspected the slowdown might instead be because of the parameters being attempted by pymc3. Under certain conditions perhaps the ODE is simply much more sensitive and requires increased time resolution to solve. I ran all samples from a slow trace through scipy's odeint and timed the solutions and determined this was not the case. It could still be the case for the augmented ODE which is something I could assess next.

In the meantime, I created my own ODE class that simply doesn't compute a gradient. This forces pymc3 to pick a gradient-free sampling technique but allows me to collect a reasonably large number of samples. A few hours of runtime is sufficient for exploratory results, more for serious analysis. But it should be noted that these sampling techniques require more samples to get the same confidence in the result both because they have more auto-correlation and because they explore the space less effectively and hence are more likely to get stuck in local minima.

I tried creating manual priors for the model partially informed by the nevergrad fit. The samples showed poor convergence; each trace was reasonably stationary but not the same as all other traces. This suggests the model found multiple local minima and failed to explore beyond them in the samples allotted. The cause is likely an under-constrained model. Many combinations of parameters can explain our data resulting in "canyons"
in the solution space that our model is exploring. This is exactly the behavior we want from pymc3 and is the reason we want a probabilistic method to assess the nevergrad fit. It appears we need more information to further constrain the model.

This could come as more data, smarter priors, better error models, or further constraints on the results (e.g. following observed ratios of symptoms, deaths, etc.).

I put some initial commentary in a jupyter notebook which I exported as slides and html. The interactive plots don't work in the slides but they do in the html (and in the notebook if you run them yourself).

- https://raaperrotta.github.io/covid-binder/presentation.slides.html
- https://raaperrotta.github.io/covid-binder/presentation.html
