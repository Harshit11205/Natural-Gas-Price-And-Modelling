# 📊 Natural Gas Price Analysis & Forecasting

## 🚀 Project Overview
This project analyzes historical natural gas prices and builds a forecasting model to predict future prices using **Polynomial Regression**.

The goal is to:
- Understand price trends over time
- Visualize historical data
- Predict future natural gas prices for the next 12 months

---

## 📂 Dataset
- Source: CSV file containing:
  - `Dates` → Monthly timestamps
  - `Prices` → Natural gas prices

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas → Data manipulation
- NumPy → Numerical computations
- Matplotlib → Data visualization

---

## 🔍 Data Preprocessing
Steps performed:
1. Loaded dataset using Pandas
2. Converted `Dates` column to datetime format
3. Sorted data from oldest to newest
4. Created a time index (`t`) for modeling

## 💻Code

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
```
### 2. Importing Database
```python
df = pd.read_csv(r"C:\Users\harsh\Downloads\Nat_Gas.csv")
------------------- Reading Database --------------------
df.head() -- Read First 5 rows of Database
df.tail() -- Read Last 5 rows of Database
df.info() -- Database Information
```
### 3. Converting Date to DateTime
``` python
df["Dates"] = pd.to_datetime(df["Dates"])
```
### 4. Sort Date From Oldest To Newest
```python
df = df.sort_values("Dates")
```
### 5. Visualising Natural Gas Prices
```python
plt.figure(figsize=(10,5))
plt.plot(df["Dates"], df["Prices"])
plt.xlabel("Date")
plt.ylabel("Natural Gas Price")
plt.title("Historical Natural Gas Prices (2020–2024)")
plt.show()
```
<img width="1068" height="591" alt="image" src="https://github.com/user-attachments/assets/966cfbc7-fc8f-462f-add4-debef40c1171" />

### 6. Converting Time To Numeric Index
```python
df["t"] = np.arange(len(df))
```
### 7. Creating A Prediction Model - Polynomial Regression Model
**Using 3 Degree because it is cubic polynomial that fits curves**

```python
coefficients = np.polyfit(df["t"], df["Prices"], 3)
```
#### Converting coefficients into a model
```python
model = np.poly1d(coefficients)
```
#### Forecasting 12 months into future
```python
future_t = np.arange(len(df), len(df)+12)
```
#### Price Prediction
```python
future_prices = model(future_t)
```
#### Generating Future Dates
```python
last_date = df["Dates"].max()
```
#### Generating Future Months
```python
future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthEnd(1),
    periods=12,
    freq="ME"
)
```
#### Creating Forecasting Table
```python
forecast_df = pd.DataFrame({
    "Dates": future_dates,
    "Prices": future_prices
})
```
#### Combining Past And Future Values
```python
combined = pd.concat([df[["Dates","Prices"]], forecast_df])
```
### 8. Comparison Between Past And Future Prices
```python
plt.figure(figsize=(10,5))
plt.plot(combined["Dates"], combined["Prices"])
plt.xlabel("Date")
plt.ylabel("Natural Gas Price")
plt.title("Natural Gas Price Forecast")
plt.show()
```
<img width="1068" height="588" alt="image" src="https://github.com/user-attachments/assets/6124acfb-2356-4edf-a06c-103457acc8cf" />
### 9. Converting Time To Numeric Index
```python
def estimate_price(date_input):

    date = pd.to_datetime(date_input)

    start_date = df["Dates"].min()

    months_passed = (date.year - start_date.year) * 12 + (date.month - start_date.month)

    price = model(months_passed)

    return float(price)
```
### 10. Estimating Prices
```python
estimate_price("2025-03-09")
```
Result - 11.217003165126675

## APPLYING STL DECOMPOSITION
#### Trend — the long-term direction of prices
#### Seasonal — repeating cycles (for gas, winter demand spikes)
#### Residual — random noise that the model cannot explain

### 1. Importing STL
```python
from statsmodels.tsa.seasonal import STL
```
### 2. Arranging Dataset Format
```python
df = pd.read_csv(r"C:\Users\harsh\Downloads\Nat_Gas.csv")

df["Dates"] = pd.to_datetime(df["Dates"])

df.set_index("Dates", inplace=True)

df = df.sort_index()
```
### 3. Applying STL Decomposition
```python
stl = STL(df["Prices"], period=12)
result = stl.fit()
```
```python
result.plot()
plt.show()
```
<img width="783" height="587" alt="image" src="https://github.com/user-attachments/assets/551e30f5-0765-4c3e-810d-6aa94bca5723" />

### THE ABOVE FOUR GRAPHS ARE AS FOLLOWS:

#### 1. Observed Data: The raw data from the market
#### 2. Trend: This shows the overall direction of prices Here trend is increasing means the market price level is increasing over time. Reasons: Inflation, Production Cost
#### 3. Seasonal: In Winter the demand usually Increases, where in Spring it Decreases, In Summers Demand is Moderate, Autumn shows as Storage Buildup.
#### 4. Residual: This represents random market shocks. Reasons: Sudden Weather events, Supply disruptions

### Seasonal Patterns In Gas Prices
```python
plt.figure(figsize=(10,4))
plt.plot(result.seasonal)
plt.title("Seasonal Pattern in Natural Gas Prices")
plt.show()
```
<img width="1038" height="462" alt="image" src="https://github.com/user-attachments/assets/d203abad-cf67-4e40-bb1f-a8f3fe669ea6" />

**Natural gas demand exhibits strong seasonal patterns driven primarily by heating demand during winter months. STL decomposition was applied to separate the time series into trend, seasonal, and residual components. The seasonal component reveals recurring annual fluctuations consistent with increased winter consumption and lower demand during warmer months.**

## Conclusion

In this project, historical natural gas price data from October 2020 to September 2024 was analyzed to understand price behavior and develop a method for estimating prices for both past and future dates. The dataset consisted of monthly market prices representing the cost of natural gas delivered at the end of each calendar month. The analysis began with data preparation, where the dataset was imported, the date column was converted into a proper datetime format, and the data was organized chronologically to ensure accurate time-series analysis.

Exploratory data analysis was conducted through visualization of the historical price series. The visualization revealed that natural gas prices exhibit fluctuations over time rather than following a simple linear pattern. These fluctuations suggest the presence of underlying trends and potential seasonal effects, which are common in energy markets due to factors such as weather conditions, changes in supply and demand, storage levels, and geopolitical events.

To model the long-term behavior of natural gas prices, a polynomial regression approach was applied. By converting the time dimension into a numerical index, a polynomial model was fitted to capture the nonlinear trend present in the data. This model was then used to extrapolate prices beyond the available dataset, allowing price estimates to be generated for future dates. The polynomial regression provided a smooth curve representing the overall trend in natural gas prices and enabled the creation of a function capable of estimating the price for any given date.

In addition to trend modeling, Seasonal-Trend Decomposition using Loess (STL) was applied to further analyze the structure of the time series. STL decomposition separates the observed price data into three components: trend, seasonal, and residual. The trend component captures the long-term movement of prices, while the seasonal component reveals recurring patterns that repeat over time, such as potential increases in demand during colder months when heating consumption rises. The residual component represents irregular fluctuations that cannot be explained by the trend or seasonal effects and may reflect unpredictable market factors.

The combination of polynomial regression and STL decomposition provided both predictive capability and interpretability. Polynomial regression enabled forecasting and price estimation, while STL decomposition offered deeper insight into the underlying structure of the data by isolating trend and seasonal patterns. Together, these techniques form a robust analytical framework for examining commodity price behavior.

Overall, this project demonstrates how time-series analysis techniques can be applied to financial and commodity market data using Python. The approach highlights the importance of data visualization, statistical modeling, and decomposition techniques in understanding market dynamics and generating price estimates. Such analytical methods are widely used in quantitative finance and energy markets to support decision-making, risk management, and pricing strategies for long-term contracts.
