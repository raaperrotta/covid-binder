# Modeling COVID-19

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/raaperrotta/covid-binder/master)

This repository houses a hodgepodge of COVID-19 related analyses. I am primarily using [nevergrad](https://facebookresearch.github.io/nevergrad/) and [pymc3](https://docs.pymc.io/) to fit models within the SEIR family. Though pymc3 offers a few different ways to handle differential equations (e.g. a scipy.ode based [DifferentialEquation](https://docs.pymc.io/notebooks/ODE_API_introduction.html?highlight=differentialequation) node and a stochastic variant, [EulerMaruyama](https://docs.pymc.io/notebooks/Euler-Maruyama_and_SDEs.html?highlight=sde)) I found the ODE integration intractible slow and needlessly precise for the slowly evolving model we consider here and found a custom implementation easier than the EulerMaruyama node given that I already had evenly spaced time steps.

My model adds noise to the observed values to represent the uncertainty surrounding them (which is equivalent to but faster than using these as observations in a noisy data model), computes the expected value of the next step for every step in the model, and compares those to the modeled next value (observed plus noise). By modifying the noise terms on the data and on the SDE transitions I can trade off the strictness of the SDE fit versus adherance to the data. Though I would prefer a more principled approach, the parameters I used are based on scrutiny of the data (to estimate realistic uncertainty terms) and the observation of some balance between deviation from the data and innovation in the model (i.e. deviation from the predicted trajectory).

Because the model is so sensitive to the model parameters (relative to my ability to guess the parameters) I was not able to get reasonable solutions with pymc3 from uninformed starting points. Instead, I used nevergrad to fit the ODE (the same model but without the noise) to the data to provide a reasonable starting point. This worked decently well but pymc3 still does not converge completely. It may be that more samples would provide the convergence I want. You can see in this [pairplot of a few parameters](https://raaperrotta.github.io/covid-binder/Italy%20COVID-19%20Model%20-%20(S)EIR%20with%20PyMC3%20and%20Nevergrad.html#3c276038-d504-4c94-9523-2cb73213a15f) that the four chains resulted in four neighboring clusters of solutions, not the single cluster you would see in an ideal situation.

The results of my work are mainly contained in a few documents, summarized here to compensate for the mess this repository has become.

### Early attempts
[This Jupyter slides presentation](https://raaperrotta.github.io/covid-binder/presentation.slides.html) from April 17th introduces the data I used for COVID-19 in the regions of Italy, an early incarnation of my pymc3 model, and some results. Here, I used the DifferentialEquation node from pymc, which was very slow and (frustratingly) became slower per iteration the more iterations it had run.

### Moving to an SDE
In [these slides](https://raaperrotta.github.io/covid-binder/presentation-2020-05-01.slides#/) from May 1st I introduce the use of the stochastic differential equation (SDE) for improved model fidelity and sampling efficiency. I also introduce a more complex model, one that separates the population into age groups, symptom levels, etc. for finer granularity of modeling. These degrees of freedom are partly constrained by observations of age distribution of cases and fatalities published by the Italian government. In this presentation, I show results of the nevergrad optimization for Lombardy, but do not attempt pymc sampling.

That pymc3 sampling proved more than a little difficult and the final results (summarized below) requierd significant simplifications of the model and the addition of lots more data (in this case by including many regions in Italy and assuming some parameters were constant across them).

### On the effectiveness of countermeasures

This is a bit of a break from the above work, which I return to below. In [slides](https://raaperrotta.github.io/covid-binder/presentation_2020_06_02.slides.html#/) and a [notebook](https://raaperrotta.github.io/covid-binder/On%20the%20effectiveness%20of%20lockdown%20measures.html) I analyze the data available from healthdata.org including their model estimates for true case count and their tabulation of enacted countermeasures to assess the relative effectiveness of each countermeasure type. See their [FAQ](http://www.healthdata.org/covid/faqs#social%20distancing) for details.

Under the assumption that none of these measures substantially effect fatality or recovery rate, we can attribute changes in the fractional rate of growth of the estimated true cases to changes in transmission rate. You can see in [these plots](https://raaperrotta.github.io/covid-binder/On%20the%20effectiveness%20of%20lockdown%20measures.html#629bc2bc-603b-4e73-9989-3aafa1978efc) the fractional growth rate colored by the status of each of the countermeasure categories. Using a pymc3 model in which each countermeasure has an independent effect on transmission rate (except stay home orders and the closure of non-essential businesses, whose interaction was separately modeled). A baseline rate was estimated per region in the healthdata.org data and a weekly periodicity estimated to account for oscillations in their estimates.

This purposefully simple model does a reasonable job of capturing the macro trends of transmission rate and unsurprisingly fails to capture exact timeseries. See for example [this model recreation of Lombardia](https://raaperrotta.github.io/covid-binder/On%20the%20effectiveness%20of%20lockdown%20measures.html#4636b84a-45b7-49f4-838a-5942f111aaa0) or [this one for Virginia](https://raaperrotta.github.io/covid-binder/On%20the%20effectiveness%20of%20lockdown%20measures.html#48e9072d-0a39-4e55-99ab-8359bc9898cd). (Note that is seems to be mostly European data that exhibits the weekly periodicity for reasons I don't yet understand.)

[The final result](https://raaperrotta.github.io/covid-binder/On%20the%20effectiveness%20of%20lockdown%20measures.html#54491e41-5fef-4f3d-9f1e-4d2c85d71282) suggests that stay at home orders are the most effective in curbing transmission, followed by closing educational facilities, followed by closing non-essential businesses. A stay at home order combined with closing non-essential businesses is more effective than either, but only slightly. Travel limitations had the least impact, perhaps due to the wide variety of measures that fall into that category.

### Pymc sampling on a simpler model

My many efforts to make pymc3 sampling work resulted in a re-parameterized and simplified model. I removed breakdown by age and symptom level and stopped tracking the susceptible population as well as those that die or recover without being detected. This limits the model populations to those tied directly to categories in the data. This in turn allowed me to parameterize the latent true traces as the observations plus noise rather than latents observed as the data through a noise model. Because my noise model was reversible, this amounts to the same thing, but is much more efficient from an implementation perspective.

I also introduced a new model element tying detection rate to the number of tests administered. See the discussion at the top of the notebook for details.

I fit all regions of Italy (except a few with very few cases overall) using nevergrad to create a good starting point for pymc3. The results can be seen in [these plots](https://raaperrotta.github.io/covid-binder/Italy%20COVID-19%20Model%20-%20(S)EIR%20with%20PyMC3%20and%20Nevergrad.html#512274b1-e8d1-446f-903d-a14f643a35f4). Then I sampled the pymc3 equivalent of that model, saw reasonable but less than ideal convergence, and analyzed the model fit. Lastly, I used the sampled parameters to forecast the [number of deaths](https://raaperrotta.github.io/covid-binder/Italy%20COVID-19%20Model%20-%20(S)EIR%20with%20PyMC3%20and%20Nevergrad.html#86843fb1-efff-4283-a4b9-864ccc135435) and [number of cases](https://raaperrotta.github.io/covid-binder/Italy%20COVID-19%20Model%20-%20(S)EIR%20with%20PyMC3%20and%20Nevergrad.html#953350c9-fa0f-4e3f-8740-589d1474a832) over the next 12 weeks. (Plots show forecasts for Lombardia (red), Emilia-Romagna (blue), and Veneto (yellow).)

----

# Raw notes:

# 24 April 2020
Perhaps I could enforce initial conditions on the Euler Maruyama SDEs using observations on the first time step. Or equally so, the final conditions or intermediate such as ratios of symptom levels.

These constraints had a huge effect! I was able to relax the parameter priors significantly and still get decent sampling. It is still not trivial to do this in a principled way but it is a promising path forward. Perhaps enough that I can remove the restrictions on relative parameters. That may even improve sampling by eliminating some funnels.

It is about time I tried this on my COVID model. TODO: I should check the prior and posterior variances and see which priors are informative. Would it be worth implementing an SDE based on something more efficient than Euler's method? I suppose as long as my population values stay positive it must be working decently well. And I can always increase the time resolution. It would be a shame to lose the speed of the SDE chasing unnecessary accuracy improvements.

## 23 April 2020
I have created an "easy" problem as an SEIR model that is fully observable and given correct, informative priors. All states are observed for the latter two thirds of the time. I can do reasonably efficient sampling on this with NUTS and SMC. MH, HMC, ADVI, and SGDV were all inaccurate or slow.

Now I am adding complexity to the model to make it more real. I have added detection and differentiated between hidden and detected cases. Only detected cases are observed. The sampling immediately showed the behavior I see in the full COVID model in which sampling gets drastically slower as it goes on. I expect the sampler is moving through the very correlated posterior space and getting stuck in funnels. Addressing this will require constraining the model and/or parameters to increase independence and identifiability.

One way would be to make stronger assumptions about the relationships of some of the parameters. I generally don't like to do that because I want the data to validate my withheld assumptions, but for sake of making the model work at all I could constrain transmission rate and death rate in detected cases to be less than in undetected cases and recovery rate greater.

Decreasing the NUTS max_tree_depth made it run faster but the results aren't great (as expected). I can see strong correlation between e0 and sigma, which makes sense as they determine how the initial scenario unfolds. Especially in the COVID model, when those values may change, we don't have a lot of data to inform these early parameters. Maybe it would be better to just run the sim while we have data and allow more states to have nonzero initial conditions.

That means adding a bunch more parameters. (I implemented it as one vector prior which means the prior is less helpful than before.) So maybe it isn't surprising that the sampling was slower than before.

To better explore the parameter space, and possible constraints and better parameterizations, I moved back to the Euler Maruyama method. This time, I reimplemented it based on the PositiveContinuous base class to ensure values are always positive (like pymc3.Lognormal).

## 22 April 2020
Most of my notes are in jupyter notebooks today. The summary:

I generated a TimeDependentEulerMaruyama based model and used nevergrad and my own euler.odeint to give it a good starting condition. I can generate samples from it but they still are finicky, often failing part way through a chain. I suspect the number of parameters is simply too high for my poorly defined model. Maybe with an easier posterior space this number of samples would not be troublesome.

I implemented a simple seir model with simulated data to explore everything I can find on ODEs in pymc. I am confident the speed of the sampling is strictly due to the sampler, not specifically the ODE portion. That said, it is almost definitely a result of a poorly defined model and not a bug in the sampler. NUTS is probably just finding it hard to generate good samples and the more it explores the harder that gets.

Something suggested in the pymc docs is sequential monte carlo. I am running a SMC sampler now on my dummy SEIR. Other possible next steps include reparameterizing the model to make parameters independent (or close to it), addressing unidentifiable parameters.

## 21 April 2020:
Stochastic differential equations! I found the EulerMaruyama distibution in the pymc3 timseries module which computes stochastic differential equation solutions as samples from a distribution where the expected values are computed using Euler's method on the ODE. It is fast!

There is a good deal of discussion in the literature about the accuracy of such a method but the general argument is that inaccuracy in the data and model outweigh inaccuracy in the ODE solver in most cases. I think that will be true here since the numbers change relatively slowly.

I had to implement my own class to add time dependence to this node. While difficult in general, the simplifying assumption that parameters are constant between time steps is well-suited to this model. I created sde_seir.TimeDependentEulerMaruyama and will write a new pymc model using it shortly.

It is running but failing with the bad initial energy error. I'll look into that tomorrow.

## 20 April 2020:
I'm taking a look at other COVID-19 models to investigate:
- possible candidates for what-if analysis
- methods worth adopting in our own model
- data sources and lessons learned using them

### healthdata.org
The COVID-19 model from IHME (see http://www.healthdata.org/covid/faqs) has pretty wide acceptance despite (and partially because of) continued scrutiny. It uses a pyhton library [Curve Fit](https://ihmeuw-msca.github.io/CurveFit/) to perform generalized logistic curves to COVID-19 hospitalization and death data. It uses numeric auto-differentiation for fast fitting without explicit gradient calculations and uses a cool-sounding iterative numeric approach for uncertainty analysis called predictive validity based uncertainty.

The Curve Fit library is open source but I haven't yet found their data pipelines or model fitting routines or fit models anywhere. They publish their predictions, which we could use, but having a fit model itself would allow us to conduct our own what-if analyses. They say in their FAQ that they are working on predictions for scenarios in which social distancing measures are lifted before the pandemic is under control (http://www.healthdata.org/covid/faqs#social%20distancing).

### covid19-scenarios.org
This analysis tool allows the user to run a COVID-19 simulation and see the results plotted against data collected from a large number of available sources. It is remarkably similar to my SEIR model work but is certainly more comprehensive, for example it models age distributions and the virus' effect on each age category differently. It appears they have dome some rudimentary fitting of their simulation to the data, perhaps only fitting against number of deaths.

This project is particularly interesting because it is entirely open source. Their data pipelines show they are using the same data for Italy we are. They also have some additional data covering things like number of hospital beds, total population, and age distribution.
- https://github.com/neherlab/covid19_scenarios/blob/master/data/parsers/italy.py
- https://github.com/neherlab/covid19_scenarios/blob/master/data/scripts/getPopulationData.py

The team responsible for this website has studied [seasonal variation in other Coronaviruses](https://smw.ch/article/doi/smw.2020.20224). The also direct users to read [this medRxiv article](https://www.medrxiv.org/content/10.1101/2020.03.04.20031112v1).

### Other
This [ESRI blog post](https://www.esri.com/about/newsroom/blog/models-maps-explore-covid-19-surges-capacity/) mentions a mobility dataset from Unacast that can show change in average distance traveled as well as change in visits to non-essential locations. This kind of individual location information is rarely available to the public but perhaps we can find an anonymized, aggregate dataset to help inform our modeling of social distancing. For example, rather than a constant rate of interaction changing at threshold times, we could base it on this data, or let it vary as a Gaussian process and infer it from this data.

Apple makes a dataset like this available: https://www.apple.com/covid19/mobility. It reports aggregates at the country level for driving, walking, and transit.

Google does too! https://www.google.com/covid19/mobility/ This is also at the country level. It aggregates data not by method of transport but by destination (retail, grocery, park, etc.) so it might be a nice complement to the Apple data.

It's not clear to me what Facebook data we can access but the descriptions suggest they might be helpful. There is [high resolution population density data](https://dataforgood.fb.com/tools/population-density-maps/) and a [region to region social connectedness index](https://dataforgood.fb.com/tools/social-connectedness-index/). This [blog post](https://about.fb.com/news/2020/04/data-for-good/) explains more about their offerings.

It appears much of this data is being funnelled to the member-only https://www.covid19mobility.org/. If the above data proves insufficient, we could consider pursuing access to this dataset.

### Kaggle
There is aways good data on Kaggle and especially for COVID-19 but it requires greater scrutiny given the open nature of its curation.

#### https://www.kaggle.com/bitsnpieces/covid19-european-enhanced-surveillance
Contains weekly case information including gender, age, outcome, hospitalization, ICU, etc.

#### https://www.kaggle.com/lanheken0/community-mobility-data-for-covid19
Expands the Google downloadable CSV by scraping the PDF reports for regional data within each country.

#### https://www.kaggle.com/davidbnn92/weather-data/output
A notebook showing how to join the [NOAA GSOD data](https://www.kaggle.com/noaa/gsod) with Kaggle's COVID-19 global forecasting data.

### Next steps
Our SEIR ODE is so slow and simple compared to the general problems for which odeint was designed. I think it is worth prusuing a simple Runge Kutta or improved Euler ODE integrator with a fixed time step over which we can easily do Bayesian inference. If we can do that, we can begin to expand this model.

Possible avenues for expansion:
- decompose population by age
- introduce climate factors
- decompose region by provinces
- introduce mobility data to inform interaction rates
- add a bin for infectious individuals with no symptoms

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
