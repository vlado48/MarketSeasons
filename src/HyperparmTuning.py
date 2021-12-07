import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from Seasonals import get_seasonals, get_ema_df, plot_emas


def score_seasonals(emas):
    '''
    Scoring functions that accepts pd.DataFrame and return avg MAPE over years
    '''
    # Scale all the whole data set to -1/1 range
    scaler = MinMaxScaler((-1, 1))   
    original_shape = emas.shape
    scaled = scaler.fit_transform(emas.to_numpy().flatten().reshape(-1, 1))
    scaled = pd.DataFrame(scaled.reshape(original_shape),
                          columns=emas.columns)

    # Calculate MAPE        
    yearly_mape = []
    for i, col in enumerate(scaled):         
        if i == 0: continue    # Skip first column
        diff = (scaled.iloc[:, i] - scaled.iloc[:, i-1]) / scaled.iloc[:, i]
        diff = diff.abs()
        MAPE = diff.mean()
        yearly_mape.append(MAPE)
    
    # Turn to ndarray and remove outliers (2008 crash)
    yearly_mape = np.array(yearly_mape)
    mean = yearly_mape.mean()
    for i, entry in enumerate(yearly_mape):
        if entry > 10 * mean:
            print(f' Deleting MAPE of {entry:.2f} that is much higher than ',
                  f'\navg {mean:.2f}. Year: {scaled.columns[i]}')
            yearly_mape = np.delete(yearly_mape, i)
            
    # Return total error for all other years        
    return yearly_mape.mean()     
    
def param_iterate(params):
    '''Returns a combinations of all parameters to grid search through'''
    keys = params.keys()
    vals = params.values()
    combinations = list(product(*vals))
    for i, c in enumerate(combinations):
        combinations[i] = dict(zip(keys, c))
    return combinations      

def grid_search(params, emas_dataframe):
    '''Conducts grid search and return the best scored set of parameters'''         
    params_combinations = param_iterate(params)
    total_iters = len(params_combinations)
    scores = np.zeros(total_iters)
    for i, c in enumerate(params_combinations):
        print(i+1,'/',total_iters)
        seasonals = get_seasonals(data,**c)    
        emas = get_ema_df(seasonals)
        score = score_seasonals(emas)
        scores[i] = score
    
    best_score_idx = scores.argmin()
    best_params = params_combinations[best_score_idx]
    print(f'Parameters giving best score of {scores.min()} are: ',best_params)
    
    return best_params

if __name__ == "__main__":
    
    # Open instrument's cleaned historical data dataframe
    instrument_name = 'usdx'
    data = pd.read_csv(f'data\\{instrument_name}_historical_tidy.csv')
    
    # Set of parmeter ranges to grid search through
    period = data.shape[0]
    window_min = period+1 if (period+1)%2==1 else period+2
    params = {'period'       : [period],
              'seasonal'     : list(range(7, 56, 4)),
              'trend'        : list(range(window_min, window_min+101, 20)),
              'robust'       : [False, True],
              'low_pass'     : list(range(window_min, window_min+101, 20)),
              'seasonal_deg' : [0, 1],
              'trend_deg'    : [0, 1],
              'low_pass_deg' : [0, 1]}
                               
    benchmark_params = {'period':data.shape[0],
                        'seasonal':11,
                        'robust':False}
    
    # Conduct grid search
    best_params = grid_search(params, data)

    # Comparison vs benchmark params   
    seasonals_benchmark = get_seasonals(data,**benchmark_params)   
    seasonals_optimized = get_seasonals(data,**best_params)    
    
    emas_benchmark = get_ema_df(seasonals_benchmark)
    plot_emas(emas_benchmark)
    
    emas_optimized = get_ema_df(seasonals_optimized)
    plot_emas(emas_optimized)
    
    print(f'Benchmark score: {score_seasonals(emas_benchmark)}',
          f'\nOptimized score: {score_seasonals(emas_optimized)}')