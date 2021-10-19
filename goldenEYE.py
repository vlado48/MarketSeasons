import csv
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import scipy.stats as stats
import os

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
    #plt.plot(prediction)
    #plt.plot(data)
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


def mutualcorel(inp_data, mode):
    corels = []
    # Return correlation sum for each dataset with other datasets
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
    
    # Return total sum of correlations
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


# correlation of yearly prices and calculated seasonal tendencies
def mutualcorel2(seasonal, prices):
    results = []
    for i, data in enumerate(prices):
        res = list(corel(data, seasonal))
        results.append(res)    
#    plt.figure()
#    plt.plot(results)
    pozit = []    
    for i in range (3):                     #For each type of calculated correlation 
        pos = 0
        for k, res in enumerate(results):   # Iterate all years results for given correlation type
            if res[i]>0:
                pos += 1              # Count each positive correlation
            if k == 0:      
                best = [res[i], k]    # First iterable set as best
            elif res[i] > best[0]:
                best = [res[i], k]    # Become best if better
        pozit.append(pos)        
        results[best[1]][i] /= 2        #devides largest correlation
        
    p, s, r = zip(*results)
    p = sum(p)/len(prices)*((pozit[0])/len(prices))
    s = sum(s)/len(prices)*((pozit[1])/len(prices))
    r = sum(r)/len(prices)*((pozit[2])/len(prices))

    #return p, s, r
    return pozit[0], pozit[1], pozit[2], 

def average(data, alfa, beta):
    prices_debased = []
    for i, price in enumerate(data):
        d = debase(price, 'add', True, alfa, beta)
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

#%% Optimize based on TS and all years total corellation
def optimize(prices, alpha_steps, alpha_range, beta_steps, beta_range):
    for i in range(alpha_steps):
        alpha = alpha_range[0] + i * (alpha_range[1]-alpha_range[0])/(alpha_steps-1)
        for j in range(beta_steps):
            beta = beta_range[0] + j * (beta_range[1]-beta_range[0])/(beta_steps-1)         
            avg = average(prices, alpha, beta)
            if i == 0 and j==0:
                b1, b2, b3 = mutualcorel2(avg, prices)
                s1 = s2 = s3 = [alpha, beta, trend, dampen]
            else:
                r1, r2, r3 = mutualcorel2(avg, prices)         
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

optimize(prices, 20, (0, 0.25), 20, (0, 0.25))

#%% Optimal values based on TS and all prices corellation

plt.ion()
#plt.plot(normalize(average(prices[:10], 0.013157894736842105, 0.02631578947368421)))
#plt.plot(normalize(average(prices[10:15], 0.013157894736842105, 0.02631578947368421)))
plt.plot(normalize(average(prices[15:], 0.013157894736842105, 0.02631578947368421)))

#%% Optimisation based on predictive power (5Y ST applied to a year ahead)
def walkforward(prices, lookback, lookforward, alpha_steps, alpha_range, beta_steps, beta_range):
    this_run = []
    for i in range(alpha_steps):
        alpha = alpha_range[0] + i * (alpha_range[1]-alpha_range[0])/(alpha_steps-1)
        subresults = []
        for j in range(beta_steps):
            beta = beta_range[0] + j * (beta_range[1]-beta_range[0])/(beta_steps-1)                     
            this_set = []
            for k in range(len(prices)-lookback-lookforward+1):       #For all years possible to lookforward w given lookback
                avg = average(prices[k:k+lookback], alpha, beta)
                result = 0
                for l in range(lookforward):                # For the number of lookforward years
                    result += np.array(corel(prices[k+lookback+l], avg))
                this_set.append(result[1])                  # Only use second correlation score
            subresults.append(np.array(this_set))
        this_run.append(np.array(subresults))
    return np.array(this_run)


#%% Score the walk forwards results
def walk_score(data, method):
    score = []
    if method == 'sum':    #average
        for run in data:
            b_score = []
            for sub in run:
                b_score.append(np.sum(sub))
            score.append(np.array([b/len(sub) for b in b_score]))
        return np.array(score)
    
    if method == 'pos':    #num of positive correlation scores
        for run in data:
            b_score = []
            for sub in run:
                pos = 0
                for val in sub:
                    if val > 0:
                        pos += 1                
                b_score.append(pos/len(sub))
            score.append(np.array(b_score))
        return np.array(score)
    
    if method == 'neg':    #average negative correlation (1-neg) higher better
        for run in data:
            b_score = []
            for sub in run:
                neg_sum = 0
                for val in sub:
                    if val < 0:
                        neg_sum += val  
                b_score.append(1 + neg_sum/len(sub))
            score.append(b_score)
        return np.array(score)
#%%
optimr = walkforward(prices, 5, 2, 10, (0, 1), 10, (0, 1))

print(optimr.shape)
#%%

a = walk_score(optimr, 'sum')
b = walk_score(optimr, 'pos')
c = walk_score(optimr, 'neg')

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.figure()
plt.imshow(b, cmap='hot', interpolation='nearest')
plt.figure()
plt.imshow(c, cmap='hot', interpolation='nearest')


#%% Optimising the LB, LF periods
def walk_optim(prices, lbrange, lfrange, alpha_steps, alpha_range, beta_steps, beta_range):
    for i in range(lbrange[0], lbrange[1]+1):
        lb = lbrange[0] + i * (alpha_range[1]-alpha_range[0])/(alpha_steps-1)
        for j in range(lfrange[0], lfrange[1]+1):