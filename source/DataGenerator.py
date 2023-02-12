import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from datetime import datetime, timedelta


class SingleTimeSeries():
    
    def __init__(
        self, 
        date_start: datetime,
        date_end: datetime,
        initial_value: float,
        random_walk_stddev: float,
        sampling_stddev: float,
        random_seed_state: int = 21
    ):
        
        self.date_start = date_start
        self.date_end = date_end
        self.initial_value = initial_value
        self.random_walk_stddev = random_walk_stddev
        self.sampling_stddev = sampling_stddev
        self.time_series_data = None
        
        self.dates = pd.date_range(start=self.date_start, end=self.date_end).tolist()
        self.number_of_days = (date_end - date_start).days + 1
        self.random_seed = np.random.default_rng(random_seed_state)
        
    
    @staticmethod
    def sample(expected_value: float, scale: float):
        raise NotImplementedError
    
        
    def generate_random_walk(self):
        random_walk = np.cumsum(self.random_seed.normal(0, self.random_walk_stddev, self.number_of_days))
        return (random_walk + self.initial_value)

    @staticmethod
    def generate_sampling_var(expected_value: np.array, scale: float):
        raise NotImplementedError
    
    
    def generate_time_series(self):
                
        unknown_value = self.generate_random_walk()
        sampled_value = self.sample(unknown_value, self.sampling_stddev)
        sampling_stddev = self.generate_sampling_var(unknown_value, self.sampling_stddev)
        self.time_series_data = pd.DataFrame(
            {'ts': self.dates, 'unknown_value': unknown_value, 'sampled_value': sampled_value, 'sampling_var': sampling_stddev}, 
            columns=['ts', 'unknown_value', 'sampled_value', 'sampling_var']
        )
        
        
    def add_trend(self, trend_value: float, date_start: datetime, date_end: datetime):
        """
        Adds a trend to the time series, both to the unknown_value and to the sampled_value
        The trend starts on date_start and finishes on date_end, when it stabilizes
        """
        
        periods = (date_end - date_start).days 
        
        thrend_values = np.full(self.number_of_days, trend_value)
        estabilization_values = np.full(self.number_of_days, trend_value*periods)
        
        trend_days = ((self.time_series_data['ts'] > date_start) & (self.time_series_data['ts'] <= date_end))
        estabilization_days = (self.time_series_data['ts'] > date_end)
        
        to_add_values = np.cumsum(thrend_values * trend_days) * trend_days + estabilization_values * estabilization_days
        
        self.time_series_data['unknown_value'] = self.time_series_data['unknown_value'] + to_add_values 
        self.time_series_data['sampled_value'] = self.time_series_data['sampled_value'] + to_add_values 
    
    
    def add_spike(self, spike_value: float, date: datetime):
        """
        Add the spike_value on the date only for the [sampled_value] column
        """
        
        to_add_values = (self.time_series_data['ts'] == date) * spike_value
        self.time_series_data['sampled_value'] = self.time_series_data['sampled_value'] + to_add_values
        

        
class SingleGaussianTimeSeries(SingleTimeSeries):

    @staticmethod
    def generate_sampling_var(expected_value: np.array, scale: float):
        return np.full(len(expected_value), self.sampling_stddev**2)

    def sample(self, expected_value: float, scale: float):
        return self.random_seed.normal(expected_value, scale)
    
    
class SingleLognormalTimeSeries(SingleTimeSeries):
    
    @staticmethod
    def generate_sampling_var(expected_value: np.array, scale: float):
        adapted_mean = np.log(expected_value) - 0.5*scale**2
        return (np.exp(scale**2) - 1) * np.exp(2*adapted_mean + scale**2)
    

    def sample(self, expected_value: float, scale: float):
        return self.random_seed.lognormal(np.log(expected_value) - 0.5*scale**2, scale)
    


class MultGaussianTimeSeries():
    
    
    def __init__(
        self, 
        date_start: datetime,
        date_end: datetime,
        initial_values: np.array,
        random_walk_cov_matrix: np.ndarray,
        sampling_stddevs: np.array,
        random_seed_state: float = 0
    ):
        
        self.date_start = date_start
        self.date_end = date_end
        self.initial_values = np.array(initial_values)
        self.random_walk_cov_matrix = random_walk_cov_matrix
        self.sampling_stddevs = np.array(sampling_stddevs)
        self.time_series_data = None
        
        self.dates = pd.date_range(start=self.date_start, end=self.date_end).tolist()
        self.number_of_days = (date_end - date_start).days + 1
        self.number_of_series = len(initial_values)
        self.random_seed = np.random.default_rng(random_seed_state)
        
        
    def sample(self, expected_values: float, scales: float):
        return self.random_seed.normal(expected_values, scales)
    
        
    def generate_random_walk(self):
        
        random_walk_steps = self.random_seed.multivariate_normal(np.full(len(self.initial_values), 0), self.random_walk_cov_matrix, self.number_of_days)
        random_walk = np.cumsum(random_walk_steps.transpose(), axis=1)
        return (self.initial_values + random_walk.transpose()).transpose()
    
    
    def generate_time_series(self):
                
        dates = np.array(list(itertools.repeat(self.dates, self.number_of_series))).flatten()
        series_names = np.array([list(itertools.repeat(f'{i}', self.number_of_days)) for i in range(self.number_of_series)]).flatten()
        unknown_value = self.generate_random_walk().flatten()
        sampling_stddev = np.array([list(itertools.repeat(stddev, self.number_of_days)) for stddev in self.sampling_stddevs]).flatten()
        sampled_value = self.sample(unknown_value, sampling_stddev)

        self.time_series_data = pd.DataFrame(
            {'ts': dates,  'serie_name': series_names, 'sampling_stddedv': sampling_stddev, 'unknown_value': unknown_value, 'sampled_value': sampled_value}, 
            columns=['ts',  'serie_name', 'sampling_stddedv', 'unknown_value', 'sampled_value']
        )
        

class MultLognormalTimeSeries(MultGaussianTimeSeries):
    
    def sample(self, expected_values: float, scales: float):
        adjusted_averages = np.log(expected_values) - scales**2/2
        return self.random_seed.lognormal(adjusted_averages, scales)
