import csv
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    instrument_name = 'usdx'
    header_rows = 16
    
    # Before creating the dataframe it will be easier to clean the data in a []
    with open('..\data\\usdx_historical.csv') as file_raw:
        data_raw = csv.reader(file_raw, delimiter=',')
        data_raw = list(data_raw)
    data_raw = data_raw[header_rows+1:]    # Remove header
    
    # Split data by year into 2 dimensional list by [year][day]. 
    yearly_split = []
    year_start = 0
    last_idx = len(data_raw) - 1
    for i, row in enumerate(data_raw):
        row[0] = dt.date.fromisoformat(row[0])         # Date to datetime object
        if i != 0:                                     # Avoid index OOR error
            if row[0].year != data_raw[i-1][0].year:   # If year changed
                yearly_split.append(data_raw[year_start:i])
                year_start = i
    
    # We need to align all the years around some date for consistency. Middle
    # off the year (1. July) will be our centering date from which each year
    # must have same amount of days prior and after.
    
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
                limits[0] = days_before    # Least days prior midyear in all data
                shortest_years[0] = (i)    # IDX of the shortest year
                
            days_after = len(year[midyear_idx[i]+1:])
            if days_after < limits[1]:
                limits[1] = days_after     # Least days after midyear in all data
                shortest_years[1] = (i)    # IDX of the shortest year
        
        # If some of the shortest years in dataset are much shorter than 
        # maximum  (~130), We need to ommit it from the dataset
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
    
    
    # Create dataframe with appropriate column names
    columns = np.arange(history_range[0], history_range[1]+1)
    data = pd.DataFrame(data=yearly_split.T, columns=columns)
    print(data.head(5))
    print('Total shape of dataframe: ', data.shape)
    data.plot()
    plt.xlabel('Trade day of a year')
    plt.ylabel('Price USD')
    
    data.to_csv(f'..\data\\{instrument_name}_historical_tidy.csv')




