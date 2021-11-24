import csv
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL


# Before creating dataframe it will be easier to align data in lists/np in a 
# way that each year has same amount of trading days and that they are syncho-
# zed
with open('euro-fx-futures.csv') as file_raw:
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

# Find minimal N of days before and after midyear across all the years
shortest_1 = np.inf
shortest_2 = np.inf
for i, year in enumerate(yearly_split):
    days_before = len(year[0:midyear_idx[i]])
    if days_before < shortest_1:
        shortest_1 = days_before
        
    days_after = len(year[midyear_idx[i]+1:])
    if days_after < shortest_2:
        shortest_2 = days_after
        
# Cut the lengths of all the longer years, separate timestamps
for i, year in enumerate(yearly_split):
    start_idx = midyear_idx[i] - shortest_1
    end_idx = midyear_idx[i] + shortest_2 + 1
    yearly_split[i] = year[start_idx : end_idx]
yearly_split = np.array(yearly_split)    # Turn into np array  

dates = yearly_split[:, :, 0]
yearly_split = yearly_split[:, :, 1].astype(float)
data_array = (yearly_split.flatten())

# We have dataset of 21 years, each having 248 trading days  
columns = np.arange(1999, 2020)
data = pd.DataFrame(data=yearly_split.T, columns=columns)
print(data.head(5))
print('Total shape of dataframe: ', data.shape)
data.plot()
plt.xlabel('Trade day of a year')
plt.ylabel('Price USD')

#%% Conduct seasonal and trend decomposition. Parameters are currently based on
#   domain knowledge and empirical testing.
stl = STL(data_array, period=249, seasonal=11, robust=False,
                 trend=251)
result_stl = stl.fit()
result_stl.plot()
seasonals_array = result_stl.seasonal.reshape((21, 248))
seasonals = pd.DataFrame(data=seasonals_array.T, columns=columns)

#%% To have a statistically more reliable seasonal dependencies we can average
#   them. However to capture slow macroeconomic changes we may want to average
#   only past few years with more weigth given to more recent years.
#   Create exponential moving average from yearly seasonal tendencies
def get_ema_df(data, order=5, alpha=0.08):
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
            print(col, ' ', coeffs[j])
            ema += data[col] * coeffs[j]    # Add up weighted years
        emas[i] = ema                       # Save as col in np array
    emas = pd.DataFrame(emas.T, columns=data.columns[order-1:])
    return emas

# Get exponential average of seasonal tendecies and plot them to see change 
# over time
def plot_emas(emas):
    for i, col in enumerate(emas):
        n_cols = emas.shape[1]   
        n_rows = data.shape[0]        
        color = 'C1' if i != n_cols-1 else 'red'
        emas[col].plot(color=color, alpha= 0.75**(n_cols-i))
        
    # Calculate relative month starts
    month = n_rows // 12
    leftover_days = n_rows - month * 12
    print(leftover_days)
    leftover = n_rows % 12
    period = np.abs(leftover - 1) // leftover + \
            (2 if np.abs(leftover - 1) // leftover == 0 else 1)
    
    lbl_idx = [(i+1)*month + ((i+period)//period) + 1 for i in range(12)]
    lbl_idx.pop()
    lbl_idx.insert(0, 0)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
              'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(lbl_idx, months)    
    
emas = get_ema_df(seasonals)
plot_emas(emas)        

#%%
leftover = 0.8
print(np.abs(leftover - 1) // leftover + (2 if np.abs(leftover - 1) // leftover == 0 else 1))





