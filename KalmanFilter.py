import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class KalmanFilterProcess():
    """
    Simple implementation of a Kalman FIlter with no trends
    The Kalman Filter basically consists of two steps the update and the prediction
    
    With the prediction, we estimate the position, together with its associated uncertainty, of the object 
    N steps in the future. As we don't have any trends, the estimate is constant and equal to its current one.
    
    With the update, the Kalman Filter updates the estimate of its current position based on the an observation.
    """
    
    def __init__(self, prior_value:float=0.0, prior_variance: float=1.0, time_variance: float=1.0):
        
        self.prior_value = prior_value
        self.prior_variance = prior_variance
        self.time_variance = time_variance
        self.prior_timestamp = None
        self.prior_date = None
        
    def update(self, observation_date: datetime, observation_value:float, observation_variance:float):
        
        update_weight = self.prior_variance / (self.prior_variance + observation_variance)
        self.prior_value = update_weight * observation_value + (1 - update_weight) * self.prior_value
        self.prior_variance = observation_variance * self.prior_variance /(observation_variance + self.prior_variance)
        self.prior_date = observation_date
        
        
    def predict(self, date : datetime):

        mean = self.prior_value
        variance = self.prior_variance + self.time_variance * (date - self.prior_date).days
        return mean, variance

    
class KalmanFilterModel():
    
    
    def __init__(
        self, 
        prior_value:float=0.0, 
        prior_variance:float=100000.0, 
        time_variance: float=1.0,
        sampling_variance: float=1.0,
        value_col_name:str="value"
        ):
        
        self.df = None
        self.ts_col_name = "ts"
        self.var_col_name = "sampling_var"
        self.value_col_name = value_col_name
        self.updated_value_col_name = "updated_value"
        self.forecast_col_name = "forecast_value"
        self.required_columns = [self.ts_col_name, self.var_col_name, self.value_col_name]
        
        self.prior_value = prior_value
        self.prior_variance = prior_variance
        self.time_variance = time_variance
        self.sampling_variance = sampling_variance
        
        self.output_df = None
        self.series_defining_columns = []
        
    def apply_kf(self, df: pd.DataFrame, forecast_date_col_name: str=None) -> pd.Series:
        """
        Apply kalman filter assuming that the whole dataframe is a single time-series
        """
        kalman_filter = KalmanFilterProcess(self.prior_value, self.prior_variance, self.time_variance)
        df = df.sort_values(self.ts_col_name)
        
        updated_values = []
        forecasted_values = []
        for index, row in df.iterrows():
            
            obs_date = row[self.ts_col_name]
            obs_value = row[self.value_col_name]
            obs_var = row[self.var_col_name]
            # obs_var = self.sampling_variance

            # Predict the position of next timestamp
            if kalman_filter.prior_date:
                kalman_filter.prior_value, kalman_filter.prior_variance = kalman_filter.predict(obs_date) 
            # Update based on most recent observation
            kalman_filter.update(obs_date, obs_value, obs_var)
            updated_values.append(kalman_filter.prior_value)
            
            if forecast_date_col_name:
                forecast_value, _ =  kalman_filter.predict(row[forecast_date_col_name])
                forecasted_values.append(forecast_value)
                
        df[self.updated_value_col_name] = updated_values
        if forecast_date_col_name:
            df[self.forecast_col_name] = forecasted_values
        
        return df
        
        
    def fit(self, df: pd.DataFrame, y:pd.Series=None):
        """
        Input:
            df: pd.DataFrame containing the timestamp, value, and variance of observation
        
        Output:
            output_df: pd.DataFrame
        """
        self.series_defining_columns = list(set(df.columns) - set(self.required_columns))
        self.output_df = df.groupby(self.series_defining_columns).apply(self.apply_kf)
    
    def predict(self, df: pd.DataFrame=None):
        raise NotImplementedError("This function was not implemented")
    