# MarketSeasons
Recurring economic activity during a calendar year impacts demant for various assets.
Foreign investments, industry restocking, agricultural or turistic seasons affect macroeconomic flows
which can be seen as a factor (one of many, or as chaos theory suggest one of np.inf) in market prices.

Here I use modern methods (TSL) for time series decomposition and try to extract insights that may provide valuable
market tendencies that should be considered by investors.

### Decompositions into components
Time signal decomposed into Trend + Seasonal + Error

<img src="https://user-images.githubusercontent.com/34378363/143443468-98634d4d-a94d-4a51-8138-e90ff3a850a7.png" width=50% height=50%>

As you can see there is a slow change in character of seasonal tendency at around year 2008 where we had a market collapse and a economic crisis.

### Seasonal tendency changes over time
Long, macroeconomic cycles can have impact on how value moves throughout economy and thus affect reccuring yearly flows.
Changes in a seasonal tendencies are averaged (Yearly periods are overlayed and averaged) by EMA (exponential moving average period=5, alpha=0.1) which keeps information up to date while
maintaining statistical strength.

<img src="https://user-images.githubusercontent.com/34378363/143444474-58c29333-f9a0-4ead-abf2-28816aa6e095.png" width=50% height=50%>

### Hyperparameter tuning
TSL allows multiple model and regression parameters that affect the results drastically. In hyperparameter optimization criterion has been poustulated in which best score is attained the lower is the MAPE (Mean absolute percentage error) of Seasonal EMA prediction VS the actual seasonal component for the predicted year. This in results provides parameters that decompose only the most constant and consistent seasonal components in the gived financial instrument. 

Seasonal component is obviously much more consistent

<img src="https://user-images.githubusercontent.com/34378363/144416709-d2d22e74-fadd-4d07-bcd7-e4e97ee6612e.png" width=50% height=50%>

We can see that this is a static component of a seasonal tendency and does not change much over the years. Score from 45% to 6% average % cumulative yearly error

<img src="https://user-images.githubusercontent.com/34378363/144416724-49e06906-8ab5-4891-9982-d714b12d2164.png" width=50% height=50%>



