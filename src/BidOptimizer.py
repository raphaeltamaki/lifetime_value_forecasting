import numpy as np
import pandas as pd
from src.VolumeAcquisition import BaseVolume
from src.ProfitModel import StandardModel
from typing import List
from itertools import product


class StandardBidOptimizer():

    def __init__(
        self,
        volume_acquisition_model: BaseVolume,
        std_values: List[float],
        ltv_fractions: List[float],
        sample_size: int=10000
        ) -> None:
        """
        volume_acquisition_model: BaseVolume variable that describes how does volume vary by cost per install?
        std_deviation_values: list of floats defining the standard deviations of the errors of the LTV predictions. These values should be normalized by the average LTV (i.e. how many fractions of the LTV is the standard deviation?) 
        ltv_fractions: list of the fractions of the average predicted LTV we should target the CPI to be
        sample_size: number of samples to be simulated for each combination of standard deviation and fraction of optimal average predicted LTV
        """


        self.volume_acquisition_model = volume_acquisition_model
        self.std_values = std_values
        self.ltv_fractions = np.array(ltv_fractions)
        self.sample_size = sample_size
        self.results = None
        self.reference_ltv_value = 1.0
        self.profit_model = StandardModel(self.volume_acquisition_model, self.reference_ltv_value)
        self.bidding_strategy_data = None


    def simulate(self):
        """
        For each of the defined values of standard deviation of the error of the lifetime value prediction, 
        simulate bidding different fractions of the LTV and evaluate the resulting profit obtained
        """

        for sd in self.std_values:

            ltv_sample = np.random.normal(self.reference_ltv_value, sd, self.sample_size)

            df = pd.DataFrame(list(product(ltv_sample, self.ltv_fractions)), columns=['estimated_ltv', 'ltv_fraction'])
            df['cpi'] = df['estimated_ltv'] * df['ltv_fraction'] 
            df['profit'] = df.apply(lambda x: self.profit_model.calculate_profit(x['cpi'], self.reference_ltv_value), axis=1)
            df['sd'] = sd
            self.results = pd.concat([self.results, df]) if self.results is not None else df


    def calculate_bidding_strategy_results(self):
        """
        From the simulations obtained from the method simulate(), calculate the average profit obtained by each pair (ltv_fraction, ltv error standard deviation).
        Then calculate for each (ltv error standard deviation) which (ltv_fraction) has the highest average profit, and what it is 
        """

        def f(x):
            output = {}
            output['mean_profit'] = np.mean(x['profit'])
            output['std_profit'] = np.std(x['profit'])
            return pd.Series(output)

        self.bidding_strategy_data = self.results.groupby(['sd', 'ltv_fraction']).apply(lambda x: f(x)).reset_index()
        self.bidding_strategy_data = self.bidding_strategy_data.sort_values('mean_profit', ascending=False)
        self.bidding_strategy_data = self.bidding_strategy_data.groupby('sd')['mean_profit', 'std_profit', 'ltv_fraction'].first().reset_index()
        self.bidding_strategy_data['ltv_fraction'] = self.bidding_strategy_data['ltv_fraction'] /  self.profit_model.optimal_cpi(self.reference_ltv_value)

    
    def run(self):
        """
        TODO
        """

        self.simulate()
        self.calculate_bidding_strategy_results()