import csv
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler


# Before creating dataframe it will be easier to align data in lists/np in a 
# way that each year has same amount of trading days and that they are syncho-
# zed
with open('data\\usdx_historical.csv') as file_raw:
    data_raw = csv.reader(file_raw, delimiter=',')
    data_raw = list(data_raw)
data_raw = data_raw[17:]    # Remove header

# Split data by year into 2 dimensional np array by year. Only full years are
# considered
yearly_split = []
year_start = 0
last_idx = len(data_raw) - 1
for i, row in enumerate(data_raw):
    row[0] = dt.date.fromisoformat(row[0])         # Date to datetime object
    if i != 0:                                     # Avoid index OOR error
        if row[0].year != data_raw[i-1][0].year:   # Current entry is new year
            yearly_split.append(data_raw[year_start:i])
            year_start = i

# We need to align all the years around some date for consistency. Middle of
# the year (1. July) will be our centering date from which each year must have 
# same amount of days prior and after.

# For each year find the index closest to 1. July (centering date)
midyear_idx = []
for year in yearly_split: 
    midyear = dt.date(year[0][0].year, 7, 1)    # Define 1.7 of given year
    midyear_d = dt.timedelta(days=365)          # Define max time delta
    closest = None
    for i, day in enumerate(year):              # For each day update delta
       if abs(day[0]-midyear) < midyear_d:      # if less than before
           midyear_d = abs(day[0]-midyear)
           index = i
    midyear_idx.append(index)

# Find minimum num. of days before and after midyear across all the years
# the maximum should be (~130)
def get_year_range(yearly_split):
    print('new fun')
    limits = [np.inf, np.inf]          # N of days before/after midyear
    shortest_years = [0, 0]            # idx of shortest year/years
    for i, year in enumerate(yearly_split):
        days_before = len(year[0:midyear_idx[i]])
        if days_before < limits[0]:
            limits[0] = days_before    # Least days period midyear across data
            shortest_years[0] = (i)    # IDX of the shortest year
            
        days_after = len(year[midyear_idx[i]+1:])
        if days_after < limits[1]:
            limits[1] = days_after     # Least days after midyear across data
            shortest_years[1] = (i)    # IDX of the shortest year
    
    # If some of the shortest years in dataset are much shorter than maximum 
    # (~130), We need to ommit it from the dataset
    for i, days in enumerate(limits):
        if days < 120:
            print(f'Year {yearly_split[shortest_years[i]][0][0].year} only',
                  f' has {days} trading days before/after the midyear',
                  '\n    - deleting from the dataset')
            yearly_split.pop(shortest_years[i])      # Delete year too short
            midyear_idx.pop(shortest_years[i])       # Delete it's IDX 
            limits = get_year_range(yearly_split)    # Recursive function call
            break

    return limits

yearly_range = get_year_range(yearly_split)

# Cut the lengths of all the longer years
for i, year in enumerate(yearly_split):
    start_idx = midyear_idx[i] - yearly_range[0]
    end_idx = midyear_idx[i] + yearly_range[1] + 1
    yearly_split[i] = year[start_idx : end_idx]
yearly_split = np.array(yearly_split)                # Turn into np array  

# Separate timestamps, hold first and last year of entire historical range,
# flatten all data to conduct time series decomposition
dates = yearly_split[:, :, 0]
yearly_split = yearly_split[:, :, 1].astype(float)
history_range = (dates[0][0].year, dates[-1][0].year)   
data_array = yearly_split.flatten()

# Create dataframe with appropriate column names
columns = np.arange(history_range[0], history_range[1]+1)
data = pd.DataFrame(data=yearly_split.T, columns=columns)
print(data.head(5))
print('Total shape of dataframe: ', data.shape)
data.plot()
plt.xlabel('Trade day of a year')
plt.ylabel('Price USD')

# Cache length of historical data and length of a year
n_cols = data.shape[1]
n_rows = data.shape[0]

#%% Conduct seasonal and trend decomposition. Parameters are currently based on
#   domain knowledge and empirical testing.
stl = STL(data_array, period=249, seasonal=11, robust=False,
                 trend=251)
result_stl = stl.fit()
result_stl.plot()
seasonals_array = result_stl.seasonal.reshape((n_cols, n_rows))
seasonals = pd.DataFrame(data=seasonals_array.T, columns=columns)

#%% To have a statistically more reliable seasonal dependencies we can average
#   them. However to capture slow macroeconomic changes we may want to average
#   only past few years with more weigth given to more recent years.
#   Create exponential moving average from yearly seasonal tendencies
def get_ema_df(data, order=5, alpha=0.08):
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
            print(col, ' ', coeffs[j])
            ema += data[col] * coeffs[j]    # Add up weighted years
        emas[i] = ema                       # Save as col in np array
    emas = pd.DataFrame(emas.T, columns=data.columns[order-1:])
    return emas

# Get exponential average of seasonal tendecies and plot them to see change 
# over time
def plot_emas(emas):
    n_cols = emas.shape[1]   
    n_rows = emas.shape[0]   
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
    
emas = get_ema_df(seasonals)
plot_emas(emas)        

#%% Save the Seasonal EMAs dataframe as csv
emas.to_csv("data\\usdx_seasonal_emas.csv")

#%%
def score_seasonals(emas):
    data = emas.copy()    # Work on copy
    scaler = MinMaxScaler((-1, 1))   
    # Scale all the columns to -1/1 range
    for i, col in enumerate(data):  
        data[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
    
    # Calculate MAPE        
    yearly_mape = []
    for i, col in enumerate(data):         
        if i == 0: continue    # Skip first column
        diff = (data.iloc[:, i] - data.iloc[:, i-1]) / data.iloc[:, i]
        diff = diff.abs()
        MAPE = diff.mean()
        yearly_mape.append(MAPE)
    
    # Turn to ndarray and remove outliers (2008 crash)
    yearly_mape = np.array(yearly_mape)
    mean = yearly_mape.mean()
    for i, entry in enumerate(yearly_mape):
        if entry > 10 * mean:
            print(f' Deleting MAPE of {entry:.2f} that is much higher than avg ',
                  f'\n{mean:.2f}. Year: {data.columns[i]}')
            yearly_mape = np.delete(yearly_mape, i)
            
    # Return total error for all other years        
    return yearly_mape.sum()         
   
#%%
print(score_seasonals(emas))

#%%





