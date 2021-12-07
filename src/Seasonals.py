import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


# Conduct seasonal and trend decomposition
def get_seasonals(data_frame, **kwargs):
    '''Returns a timeseries of seasonal component of original timeseries'''
    n_cols = data_frame.shape[1]
    n_rows = data_frame.shape[0]
    data_array = data_frame.to_numpy().T.flatten()
    stl = STL(data_array, **kwargs)
    result_stl = stl.fit()
    result_stl.plot()
    seasonals_array = result_stl.seasonal.reshape((n_cols, n_rows))
    seasonals = pd.DataFrame(data=seasonals_array.T,
                             columns=data_frame.columns)
    return seasonals

#   To have a statistically more reliable seasonal dependencies we can average
#   them. However to capture slow macroeconomic changes we may want to average
#   only past few years with more weigth given to more recent years.
#   Create exponential moving average from yearly seasonal tendencies
def get_ema_df(data, order=5, alpha=0.08):
    '''
    Returns a dataframe of averaged (EMA) seasonal tendencies.
    '''
    n_cols = data.shape[1]
    n_rows = data.shape[0]    
    if n_cols < order:
        raise ValueError('Data must have at least as many columns '
                         'as is the order of EMA')    
    
    # Create array of coeffecients / weigths
    coeffs = np.array([1])
    for i in range(1, order):
        coeffs = np.append(coeffs, (1 - alpha)**i)
    coeffs = np.flip(coeffs)

    # Create a df of EMAs 
    n_emas = n_cols - order + 1             # Number of EMA columns
    emas = np.zeros((n_emas, n_rows))       # Initialize 2D array for EMAs  
    for i in range(n_emas):
        ema = np.zeros(n_rows)
        for j, col in enumerate(data.iloc[:, i:(i+order)]):
            ema += data[col] * coeffs[j]    # Add up weighted years
        emas[i] = ema                       # Save as col in np array
    emas = pd.DataFrame(emas.T, columns=data.columns[order-1:])
    return emas

# Get exponential average of seasonal tendecies and plot them to see change 
# over time
def plot_emas(emas):
    '''Returns a plot showing evolution of seasonal averages in time.'''
    n_cols = emas.shape[1]   
    n_rows = emas.shape[0]   
    plt.figure()
    for i, col in enumerate(emas):    
        color = 'C1' if i != n_cols-1 else 'red'
        emas[col].plot(color=color, alpha= 0.75**(n_cols-i))
        
    # Calculate relative month starts (corresponding indexes)
    month = n_rows // 12             # Ideal month length          
    leftover = (n_rows % 12) / 12    # Days left over per month

    # Calculate per how many months an extra day should be added
    period = np.abs(leftover - 1) // leftover + \
             (2 if (np.abs(leftover - 1) // leftover == 0) else 1)

    # Calculate the indexes of month starts, assuming Jan starts with extra day
    lbl_idx = [(i+1)*month + ((i+period)//period) + 1 for i in range(12)]
    lbl_idx.pop()
    lbl_idx.insert(0, 0)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
              'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(lbl_idx, months)    


if __name__ == "__main__":
    
    # Open given instrument historical data, get seasonal with default params
    instrument_name = 'usdx'
    data = pd.read_csv(f'..\data\\{instrument_name}_historical_tidy.csv')
    
    # Conduct seasonal and trend decomposition.
    benchmark_params = {'period':data.shape[0],
                        'seasonal':11,
                        'robust':False}
    seasonals = get_seasonals(data)    
    emas = get_ema_df(seasonals)
    plot_emas(emas)    
    
    emas.to_csv(f'..\data\\{instrument_name}_seasonal_emas.csv')