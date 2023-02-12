import numpy as np
import pandas as pd
from VolumeAcquisition import BaseVolume
from scipy.optimize import minimize


class StandardModel():
    
    def __init__(self, volume_model: BaseVolume, lifetime_value: float):
        
        self.volume_model = volume_model
        self.lifetime_value = lifetime_value
        
    def calculate_profit(self, cpi: float, ltv: float=None) -> float:
        ltv = ltv or self.lifetime_value
        return self.volume_model.calculate_volume(cpi) * (ltv - cpi)
    
    def _calculate_loss(self, cpi: float) -> float:
        return -self.calculate_profit(cpi)
    
    def optimal_cpi(self, starting_value: float=1.0) -> float:
        """Find optimal bid based on the volume function using simplex"""
        return minimize(self._calculate_loss, starting_value, constraints = {'type': 'ineq', 'fun': lambda x:  x[0]}).x[0]
    
    def output_profit_data(self, cpi_values: np.array) -> pd.DataFrame:
        
        df = pd.DataFrame({'cpi': cpi_values}, columns=['cpi'])
        df['volume'] = df.apply(lambda x: self.volume_model.calculate_volume(x['cpi']), axis=1)
        df['profit'] = df.apply(lambda x: self.calculate_profit(x['cpi']), axis=1)
        return df

        
        