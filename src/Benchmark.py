import stan
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from datetime import datetime

import time
import os
import random
from sklearn.preprocessing import OneHotEncoder


def store_results(data: pd.DataFrame, path: str) -> None:
    if os.path.exists(path):
        existing_data = pd.read_csv(path)
        data = pd.concat([existing_data, data])

    data.to_csv(path)


def load_data(data_path: str):
    return pd.read_csv(data_path)


# Constants
class Benchmarker:
    def __init__(
        self,
        data_path="~/Downloads/train.csv",
        sample_sizes=[50, 100, 200, 500],
        date_start=datetime(2018, 12, 1),
        date_end=datetime(2019, 5, 1),
        n_iterations=100,
        random_state=42,
    ) -> None:
        self.data_path = data_path
        self.sample_sizes = sample_sizes
        self.date_start = date_start
        self.date_end = date_end
        self.n_iterations = n_iterations
        self.random_state = random_state

    @staticmethod
    def time_pymc_model_execution(
        data, sampler=None, use_gaussian_random_walk_class=False
    ):
        # Define the coordinates for the time series

        dates_tensor = data["join_date"].values
        target_tensor = data["target"].values
        n_days = len(data["join_date"].unique())
        coords = {"steps": data["join_date"].values}

        start_time = time.time()

        with pm.Model(coords=coords) as model:
            # Priors on Gaussian random walks
            alpha_prior = pm.Normal("alpha_prior", 4, 3, shape=1)
            alpha_dev = pm.HalfCauchy("alpha_dev", 2)
            sigma = pm.HalfCauchy("sigma", 2)

            if use_gaussian_random_walk_class:
                alpha = pm.GaussianRandomWalk(
                    "alpha",
                    mu=0,
                    sigma=alpha_dev,
                    init_dist=pm.Normal.dist(4, 3),
                    shape=n_days,
                )
            else:
                alpha = pm.Deterministic(
                    "alpha",
                    pt.concatenate(
                        [
                            alpha_prior,
                            pm.Normal("alpha_raw", sigma=alpha_dev, shape=n_days - 1),
                        ]
                    ).cumsum(axis=0),
                )

            # Likelihood
            likelihood = pm.Normal(
                "likelihood",
                mu=alpha[dates_tensor],
                sigma=sigma,
                observed=target_tensor,
                dims="steps",
            )

            # MCMC sampling
            if sampler:
                _ = pm.sample(1000, chains=3, nuts_sampler=sampler)
            else:
                _ = pm.sample(1000, chains=3)

        return time.time() - start_time

    @staticmethod
    def time_pystan_model_execution(data):
        time_series_model = (
            """
            data {
            int<lower=0> N;
            int<lower=0> n_dates;
            int date[N];
            real observation[N];
            }"""
            + (round(random.uniform(0, 100000)) * " ")
            + """ 
            parameters {
                vector[n_dates] mu; //The random-walking signal
                real<lower=0.0001> sampling_stddev; // the standard deviation of the user-level LTV
                real<lower=0.0001> random_walk_stddev; // Standard deviation of random walk
            }
            model {
                sampling_stddev ~ cauchy(0, 2); // Define the prior for the user-level LTV
                random_walk_stddev ~ cauchy(0, 2); // Define the prior for the random walk stddev
                mu[1] ~ normal(4, 3); // 
                mu[2:n_dates] ~ normal(mu[1:(n_dates - 1)], random_walk_stddev); // Vectorized

                observation ~ normal(mu[date], sampling_stddev); // Vectorized
            }
            """
        )

        # get the data for STAN
        stan_format_data = {}
        stan_format_data["N"] = int(data.shape[0])
        stan_format_data["n_dates"] = len(data["join_date"].unique())
        stan_format_data["date"] = list(
            data["join_date"].values + 1
        )  # +1 because STAN start 1
        stan_format_data["observation"] = list(data["target"].values)

        # train the model
        start_time = time.time()
        # pass the model to STAN (doesn't 'train the model' per say)
        posterior = stan.build(time_series_model, data=stan_format_data)
        # sample (is the actually 'fit')
        _ = posterior.sample(num_chains=3, num_samples=1000)

        return time.time() - start_time

    def sample_data(self, data, sample_size):
        sampeld_data = data.sample(sample_size, replace=True)
        return sampeld_data[
            (sampeld_data["join_date"] >= self.date_start)
            & (sampeld_data["join_date"] <= self.date_end)
        ]

    def data_preprocessing(self, data):
        # cast join_date to int (days since date_start) to pass it to stan
        data["join_date"] = pd.to_datetime(data["join_date"])
        data = data[
            (data["join_date"] >= self.date_start)
            & (data["join_date"] <= self.date_end)
        ]
        data["join_date"] = (data["join_date"] - self.date_start).dt.days.values
        return data

    def benchmark(self):
        data = load_data(self.data_path)
        data["join_date"] = pd.to_datetime(data["join_date"])

        for i in range(self.n_iterations):
            for sample_size in self.sample_sizes:
                sampled_data = self.sample_data(data, sample_size)
                sampled_data = self.data_preprocessing(sampled_data)

                pymc_exec_time = [self.time_pymc_model_execution(sampled_data)]
                pystan_exec_time = [self.time_pystan_model_execution(sampled_data)]
                nutpie_exec_time = [
                    self.time_pymc_model_execution(sampled_data, "nutpie")
                ]
                numpyro_exec_time = [
                    self.time_pymc_model_execution(sampled_data, "numpyro")
                ]

                results = pd.DataFrame(
                    {
                        "pymc_time": pymc_exec_time,
                        "pystan_exec_times": pystan_exec_time,
                        "nutpie_exec_times": nutpie_exec_time,
                        "numpyro_exec_times": numpyro_exec_time,
                        "it": [i],
                        "sample_size": [sample_size],
                    }
                )

                store_results(
                    results, "/Users/raphaeltamaki/Documents/pymc_pystan_results.csv"
                )

    def pymc_classes_benchmark(self):
        data = load_data(self.data_path)
        data["join_date"] = pd.to_datetime(data["join_date"])

        for i in range(self.n_iterations):
            for sample_size in self.sample_sizes:
                sampled_data = self.sample_data(data, sample_size)
                sampled_data = self.data_preprocessing(sampled_data)

                stats_exec_time = [self.time_pymc_model_execution(sampled_data)]
                grw_exec_time = [
                    self.time_pymc_model_execution(
                        sampled_data, use_gaussian_random_walk_class=True
                    )
                ]

                results = pd.DataFrame(
                    {
                        "Base Distributions": stats_exec_time,
                        "GaussianRandomWalk": grw_exec_time,
                        "it": [i],
                        "sample_size": [sample_size],
                    }
                )

                store_results(
                    results,
                    "/Users/raphaeltamaki/Documents/pymc_classes_comparison.csv",
                )


class GroupsNumberBenchmarker:
    def __init__(
        self,
        data_path="~/Downloads/train.csv",
        simulations_data_path="/Users/raphaeltamaki/Documents/pymc_pystan_groups_results.csv",
        sample_size=10000,
        date_start=datetime(2018, 12, 1),
        date_end=datetime(2019, 5, 1),
        n_iterations=100,
        random_state=42,
    ) -> None:
        self.data_path = data_path
        self.simulations_data_path = simulations_data_path
        self.sample_size = sample_size
        self.date_start = date_start
        self.date_end = date_end
        self.n_iterations = n_iterations
        self.random_state = random_state

    def sample_data(self, data, sample_size):
        sampeld_data = data.sample(sample_size, replace=True)
        return sampeld_data[
            (sampeld_data["join_date"] >= self.date_start)
            & (sampeld_data["join_date"] <= self.date_end)
        ]

    def data_preprocessing(self, data):
        # cast join_date to int (days since date_start) to pass it to stan
        data["join_date"] = pd.to_datetime(data["join_date"])
        data = data[
            (data["join_date"] >= self.date_start)
            & (data["join_date"] <= self.date_end)
        ]
        data["join_date"] = (data["join_date"] - self.date_start).dt.days.values

        # Apply OneHotEncoder so that we can use the categorical variables directly
        countries_enc = OneHotEncoder()
        card_enc = OneHotEncoder()
        product_enc = OneHotEncoder()

        encoded_countries = countries_enc.fit_transform(
            np.array(data[["country_segment"]])
        ).toarray()
        encoded_credit_card_level = card_enc.fit_transform(
            np.array(data[["credit_card_level"]])
        ).toarray()
        encoded_product = product_enc.fit_transform(
            np.array(data[["product"]])
        ).toarray()

        data[countries_enc.categories_[0]] = encoded_countries
        data[card_enc.categories_[0]] = encoded_credit_card_level
        data[product_enc.categories_[0]] = encoded_product
        features_groups = [
            list(countries_enc.categories_[0]),
            list(countries_enc.categories_[0]) + list(card_enc.categories_[0]),
            list(countries_enc.categories_[0])
            + list(card_enc.categories_[0])
            + list(product_enc.categories_[0]),
        ]

        return data, features_groups

    @staticmethod
    def time_pystan_model_execution(X: pd.DataFrame, y: pd.Series, features: list):
        time_series_model = """
            data {
                int<lower=1> N;
                int<lower=1> K;
                matrix[N,K]  X;
                vector[N]    observation;
            }
            parameters {
                vector[K] beta;
                real beta0;
                real<lower=0.0001> sigma;
            }
            model {
                beta0 ~ normal(0, 10);
                beta ~ normal(6, 10);
                sigma ~ cauchy(0, 2);
                observation ~ normal(beta0 + X * beta, sigma);
            }
        """

        # get the data for STAN
        stan_format_data = {}
        stan_format_data["N"] = X.shape[0]
        stan_format_data["K"] = len(features)
        stan_format_data["X"] = np.matrix(X[features])
        stan_format_data["observation"] = list(y)

        # train the model
        start_time = time.time()
        # pass the model to STAN (doesn't 'train the model' per say)
        posterior = stan.build(time_series_model, data=stan_format_data)
        # sample (is the actually 'fit')
        fit = posterior.sample(num_chains=3, num_samples=1000)
        return time.time() - start_time

    @staticmethod
    def time_pymc_model_execution(
        X: pd.DataFrame, y: pd.Series, features: list, sampler=None
    ):
        """
        Run PyMC model with the given data
        """
        
        # Define the coordinates for the time series
        feature_values = X[features].values
        start_time = time.time()
        with pm.Model(coords={"predictors": features}) as model:
            # Priors on Gaussian random walks
            beta0 = pm.Normal("beta0", 6, 10)
            beta = pm.Normal("beta", 6, 10, dims="predictors")
            sigma = pm.HalfCauchy("sigma", 2)

            # Likelihood
            _ = pm.Normal(
                "observation",
                mu=beta0 + pt.dot(feature_values, beta),
                sigma=sigma,
                observed=y,
            )

            # MCMC sampling
            if sampler is None:
                _ = pm.sample(1000, chains=3)
            else:
                _ = pm.sample(1000, chains=3, nuts_sampler=sampler)

        return time.time() - start_time

    def benchmark(self):
        data = load_data(self.data_path)
        data["join_date"] = pd.to_datetime(data["join_date"])

        # process the data
        sampled_data = self.sample_data(data, self.sample_size)
        sampled_data, features_groups = self.data_preprocessing(sampled_data)

        # variables to store simulations data
        pystan_times = []
        pymc_times = []
        its = []
        sample_sizes = []
        number_of_groups = []

        for i in range(self.n_iterations):
            for features in features_groups:
                pystan_times.append(
                    self.time_pystan_model_execution(
                        sampled_data, sampled_data["target"], features
                    )
                )
                pymc_times.append(
                    self.time_pymc_model_execution(
                        sampled_data, sampled_data["target"], features
                    )
                )
                its.append(i)
                sample_sizes.append(self.sample_size)
                number_of_groups.append(len(sampled_data[features].drop_duplicates()))

                results = pd.DataFrame(
                    {
                        "pymc_time": pymc_times,
                        "pystan_exec_times": pystan_times,
                        "it": its,
                        "number_of_groups": number_of_groups,
                        "sample_size": sample_sizes,
                    }
                )

                results.to_csv(self.simulations_data_path)
