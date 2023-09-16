# Lifetime Value Forecasting
Lifetime Value Forecasting for digital focused products.
The classes in this folder are meant to replicate the data commonly seen when advertising in fremium digital products, where the vast majority of users don't generate any revenue, and the users that do often follow a long-tailed distribution, almost (if not actually) resembling a pareto distribution

# PyData Amsterdam 2023 - Forecasting Customer Lifetime Value with PySTAN
In September 2023, I had the opportunity to share how to forecast Customer Lifetime Value (CLTV) using PySTAN, gave the reason why it matters to estimate distributions and not only point estimated when predicting CLTV for marketing campaigns, and compared PySTAN and PyMC in terms of computational performance.

All the content shared in the [presentation](https://docs.google.com/presentation/d/1SD2aUMpcRAStDlc4BtcvhuTlSJOlsREZAmPYjMKEnEg/edit#slide=id.g25a81e3591d_0_0) used the code available in the repository. More specifically

- [PyData Amsterdam 2023 - Forecasting Lifetime for every day in the dataframe](https://github.com/raphaeltamaki/lifetime_value_forecasting/blob/main/PyData%20Amsterdam%202023%20-%20Forecasting%20Lifetime%20for%20every%20day%20in%20the%20dataframe.ipynb): shows how to forecast LTV for each day in the dataset
- [PyData Amsterdam 2023 - Benchmar](https://github.com/raphaeltamaki/lifetime_value_forecasting/blob/main/PyData%20Amsterdam%202023%20-%20Benchmark.ipynb) and [src/Benchmark.py](https://github.com/raphaeltamaki/lifetime_value_forecasting/blob/main/src/Benchmark.py) holds the code for running the benchmark between the different implementations of a gaussian random walk in PyMC and the benchmark betwene PyMC and PySTAN
- [/results](https://github.com/raphaeltamaki/lifetime_value_forecasting/tree/main/results) folder holds the results of the benchmark above used to generate the plots for the presentation