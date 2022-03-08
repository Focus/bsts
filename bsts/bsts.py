import matplotlib.pyplot as plt

import numpy as np
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive

from .min_max_scaler import MinMaxScaler
from .horseshoe import horseshoe_prior


class BSTS(object):
    def __init__(self, seasonality):
        self.seasonality = seasonality
        self.samples = None
        self.mcmc = None
        self.y_train = None
        self.scaler = MinMaxScaler()

    def _model_fn(self, y, X=None, future=0):
        N = len(y)
        freedom = numpyro.sample('freedom', dist.Uniform(2, 20))
        scale_delta = numpyro.sample(
            'scale_delta', dist.HalfNormal(0.5 / self.seasonality)
        )
        scale_mu = numpyro.sample('scale_mu', dist.HalfNormal(5.0))
        scale_tau = numpyro.sample('scale_tau', dist.HalfNormal(5.0))
        scale_y = numpyro.sample('scale_y', dist.HalfCauchy(5.0))

        init_mu = numpyro.sample(
            'init_mu', dist.Normal(0.0, 5.0)
        )
        init_delta = numpyro.sample(
            'init_delta', dist.Normal(0.0, 10.0)
        )
        init_tau = numpyro.sample(
            'init_tau', dist.Normal(jnp.zeros(self.seasonality - 1), 5.0)
        )

        regression_term = 0.0
        if X is not None:
            beta = horseshoe_prior(X.shape[1])
            reg_constant = numpyro.sample(
                'reg_constant', dist.Normal(0.0, 10.0)
            )
            regression_term = jnp.dot(X, beta) + reg_constant

        def transition_fn(carry, t):
            tau, delta, mu = carry

            exp_tau = -tau.sum()
            new_tau = numpyro.sample('tau', dist.Normal(exp_tau, scale_tau))
            new_tau = jnp.where(t < N, new_tau, exp_tau)
            tau = jnp.concatenate([tau, new_tau[None]])[1:]

            new_delta = numpyro.sample(
                'delta', dist.Laplace(delta, scale_delta)
            )
            new_delta = jnp.where(t < N, new_delta, delta)

            exp_mu = mu + delta
            new_mu = numpyro.sample(
                'mu', dist.Normal(loc=exp_mu, scale=scale_mu)
            )
            new_mu = jnp.where(t < N, new_mu, exp_mu)

            expectation = new_mu + new_tau
            if X is not None:
                expectation += regression_term[t]
            y_model = numpyro.sample(
                'y', dist.StudentT(df=freedom, loc=expectation, scale=scale_y)
            )
            return (tau, new_delta, new_mu), y_model

        with numpyro.handlers.condition(data={'y': y}):
            _, ys = scan(
                transition_fn,
                (init_tau, init_delta, init_mu),
                jnp.arange(N + future)
            )
        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])

    def fit(self,
            y,
            X=None,
            num_warmup=2000,
            num_samples=2000,
            num_chains=4,
            rng_key=random.PRNGKey(0)):
        self.y_train = jnp.array(self.scaler.fit_transform(y))
        self.X_train = None
        if X is not None:
            assert X.shape[0] == len(y)
            self.X_train = jnp.array(self.scaler.transform(X))
        kernel = NUTS(self._model_fn)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains
        )
        self.mcmc.run(rng_key, self.y_train, self.X_train)
        self.samples = self.mcmc.get_samples()

    def predict(self, future, X=None, rng_key=random.PRNGKey(1)):
        if self.samples is None:
            raise ValueError(
                'Model must be fit before prediction.'
            )
        predictive = Predictive(self._model_fn,
                                self.samples,
                                return_sites=["y_forecast"])
        if X is not None:
            X = self.scaler.transform(X)
            X = jnp.concatenate([self.X_train, X], axis=0)
        preds = predictive(rng_key, self.y_train, X, future=future)[
            "y_forecast"
        ]
        return self.scaler.inv_transform(preds)

    def plot(self):
        if self.samples is None:
            raise ValueError(
                'Model must be fit before prediction.'
            )
        y_train = self.scaler.inv_transform(self.y_train)
        fig, axes = plt.subplots(figsize=(12, 14), nrows=4)
        axes = axes.flatten()
        axes[0].plot(y_train, label='actual')
        preds = (
            self.samples['tau'].mean(axis=0) +
            self.samples['mu'].mean(axis=0)
        )
        if self.X_train is not None:
            preds += self.samples['reg_constant'].mean()
        preds = self.scaler.inv_transform(preds)
        axes[0].plot(preds, label='predicted')
        axes[0].set_title('Time series')
        axes[0].legend()
        axes[1].plot(
            self.scaler.inv_transform(self.samples['mu'].mean(axis=0))
        )
        axes[1].set_title('Trend')
        axes[2].plot(
            self.scaler.inv_transform(self.samples['delta'].mean(axis=0))
        )
        axes[2].set_title('Change in trend')
        axes[3].plot(
            self.scaler.inv_transform(self.samples['tau'].mean(axis=0))
        )
        axes[3].set_title('Seasonality')
        return fig, axes

    def plot_future(self, y_future, X_future=None):
        forecast = self.predict(len(y_future), X_future)
        mean_forecast = forecast.mean(axis=0)
        std_forecast = forecast.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        x_train = np.arange(len(self.y_train))
        x_test = np.arange(
            len(self.y_train),
            len(self.y_train) + len(mean_forecast)
        )
        y_train = self.scaler.inv_transform(self.y_train)
        ax.plot(x_train, y_train, label='training')
        ax.plot(x_test, mean_forecast, label='prediction')
        ax.plot(x_test, y_future, label='actual', linestyle='--')
        ax.fill_between(
            x_test,
            mean_forecast - std_forecast,
            mean_forecast + std_forecast,
            color='black',
            alpha=0.2
        )
        ax.legend()
        return fig, ax
