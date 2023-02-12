import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


def simple_plot(df, date_col: str, true_value:str ='unknown_value', obs_col:str = 'value', predicted_col: str='updated_value'):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(20, 10))
    floor_line = np.linspace(0, 0, df.shape[0])
    ax = plt.axes()
    ax.plot(df[date_col], df[true_value], color = 'gray');
    ax.plot(df[date_col], df[obs_col], color = 'blue');
    ax.plot(df[date_col], df[predicted_col], color = 'blue', linestyle='dotted');
    ax.plot(df[date_col], floor_line, color = 'gray', linestyle='dotted');

def simple_line_plot(df: DataFrame, x_name: str, y_name: str):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(20, 10))
    floor_line = np.linspace(0, 0, df.shape[0])
    ax = plt.axes()
    ax.plot(df[x_name], floor_line, color = 'gray', linestyle='dotted');
    ax.plot(df[x_name], df[y_name], color = 'gray');



def multi_variable_plot(df, grouping_column='serie_name', value_column='unknown_value'):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(20, 10))
    floor_line = np.linspace(0, 0, df.shape[0])
    ax = plt.axes()
    
    ax.plot(df['ts'], floor_line, color = 'gray', linestyle='dotted')
    df.set_index('ts', inplace=True)
    df.groupby(grouping_column)[value_column].plot()
    