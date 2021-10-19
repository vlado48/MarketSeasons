import csv
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import scipy.stats as stats
import os

print(os.getcwd())

with open('euro-fx-futures.csv') as file_raw:
    file = csv.reader(file_raw, delimiter=',')
    file = list(file)

# cut header
file = file[17:]

# split data by year
yearly = []
start = 0
for i, row in enumerate(file):
    row[0] = dt.date.fromisoformat(row[0])  #Change to datetime object
    if i != 0:
        if row[0].year != file[i-1][0].year:   #See if year changed
            yearly.append(file[start:i])
            start = i

# Create list of day differences of each year wrt midyear date
midyear_idx = []
for year in yearly: 
    midyear = dt.date(year[0][0].year, 7, 1)
    midyear_d = dt.timedelta(days=365)
    closest = None
    for i, day in enumerate(year):
       if abs(day[0]-midyear) < midyear_d:
           midyear_d = abs(day[0]-midyear)
           index = i
    midyear_idx.append(index)
#%%
# Adjust arrays of each year so that they are aligned at midyear by deleting 
# first days of years that have longer first half 
midyear_min = min(midyear_idx)
for i, year in enumerate(midyear_idx):
    delta = year-midyear_min
    if delta != 0:
        for j in range(delta):
            del yearly[i][0]


shortest_yr = 365            
for year in yearly:
    if len(year) < shortest_yr:
        shortest_yr = len(year)    

for year in yearly:
    delta = len(year)-shortest_yr
    for i in range(delta):
        del year[-1]

dates = []
prices = []
for year in yearly:
    d, p = zip(*year)
    dates.append(d)
    p = list(p)
    p = [float(i) for i in p]
    prices.append(p)
    
#%%
    
# Normalizes in such manner that first datapoint is 0, has negative values
def normalize1(inp_data):
    data = list(inp_data)
    bot = min(data)
    top = max(data)
    price_range = top - bot
    normal_tick = 100 / price_range # % per pip
    zero = data[0]
    for i, val in enumerate(data):
        val = (val - zero) * normal_tick
        if val == 0:
            val += 0.01 
        data[i] = val
    return data

# Normalizes such that lowest value is 0, has no negative values
def normalize(inp_data):
    data = list(inp_data)
    bot = min(data)
    top = max(data)
    price_range = top - bot
    normal_tick = 100 / price_range # % per pip
    for i, val in enumerate(data):
        val = (val - bot) * normal_tick
        if val == 0:
            val += 0.01 
        data[i] = val
    return data

# Detrends the data based on ExponentialSmoothing model
def debase(inp_data, trend, dampen, alpha, beta):
    data = list(inp_data)
    model = ExponentialSmoothing(data, trend = trend, damped_trend=dampen, seasonal=None)
    fit = model.fit(smoothing_level = alpha, smoothing_trend = beta)   
    prediction = fit.predict(0, len(data))
    for i, val in enumerate(data):
        data[i] = data[i] - prediction[i]
    return data

# Calculates correlation goodness
def corel(data1, data2):
    c1, p1 = stats.pearsonr(data1, data2)
    c2, p2 = stats.spearmanr(data1, data2)
    c3 = stats.kendalltau(data1, data2)[0]
    p3 = stats.kendalltau(data1, data2)[1] 
    return c1*(1-p1), c2*(1-p2), c3*(1-p3)

# Rerturn array of correlations among all datasets in array
def mutualcorel(inp_data, mode):
    corels = []

    for i, data in enumerate(inp_data):
        p = s = k = 0
        for j in range(0, i):
            p_res, s_res, k_res = corel(data, inp_data[j])
            p += p_res
            s += s_res
            k += k_res
        for j in range(i, len(inp_data)):
            if j == i:
                continue
            p_res, s_res, k_res = corel(data, inp_data[j])
            p += p_res
            s += s_res
            k += k_res
        corels.append([p, s, k])
    if mode == 'sum':
        temp = [0, 0, 0]
        for year in corels:
            temp[0] += year[0]
            temp[1] += year[1]
            temp[2] += year[2]
        temp = [val/len(corels) for val in temp]
        return temp    
    else:   
        return corels


# needs fixing due to change in fitting
def average(data):
    prices_debased = []
    for i, price in enumerate(data):
        d = debase(price, 'add', True, 0.4358974358974359, 0.9743589743589743)
        n = normalize1(d)       
        prices_debased.append(n)
    for i, price in enumerate(prices_debased):
        if i != 0:
            for j, val in enumerate(price):
                debased_avg[j] += val 
        else:
            debased_avg = price
           
    debased_avg = [i/len(prices_debased) for i in debased_avg]
    return debased_avg

#%%

plt.plot(average(prices))
'''
# testing of optimisation method
trend = 'add'
dampen = 'False'
#alpha = 0.8
#beta = 0.8


prices_together = []
for price in prices:
    prices_together += price
test = debase(prices_together, 'add', True, 0.55, 0.55) 
tsaplots.plot_acf(test, lags = 480)
tsaplots.plot_pacf(test, lags = 480)    
'''
#%%
def optimize(prices, alpha_steps, alpha_range, beta_steps, beta_range):
    for i in range(alpha_steps):
        alpha = alpha_range[0] + i * (alpha_range[1]-alpha_range[0])/(alpha_steps-1)
        for j in range(beta_steps):
            beta = beta_range[0] + j * (beta_range[1]-beta_range[0])/(beta_steps-1)
            test = []
            for price in prices:
                test.append(normalize1(debase(price, trend, dampen, alpha, beta)))
            if i == 0 and j==0:
                b1, b2, b3 = mutualcorel(test, 'sum')
                s1 = s2 = s3 = [alpha, beta, trend, dampen]
            else:
                r1, r2, r3 = mutualcorel(test, 'sum')           
                if r1>b1:
                    b1 = r1
                    s1 = [alpha, beta, trend, dampen]
                if r2>b2:
                    b2 = r2
                    s2 = [alpha, beta, trend, dampen]
                if r3>b3:
                    b3 = r3
                    s3 = [alpha, beta, trend, dampen]      
    print(b1, s1, b2, s2, b3, s3)

optimize(prices, 40, (0, 1), 40, (0, 1))    

