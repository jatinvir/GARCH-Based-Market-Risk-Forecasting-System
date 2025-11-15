#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import xgboost as xgb
import json


# In[250]:


aapl = yf.download("AAPL", start="2005-01-01", group_by="column", auto_adjust=False)


# In[251]:


aapl.columns = aapl.columns.get_level_values(0)
print(aapl.columns)
aapl.tail()


# In[252]:


aapl["return"] = aapl["Adj Close"] / aapl["Adj Close"].shift(1) - 1
aapl


# In[253]:


aapl["logreturn"] = np.log(aapl["Adj Close"]).diff()
aapl


# In[254]:


print(aapl["logreturn"].mean(), aapl["logreturn"].std())


# In[255]:


plt.figure(figsize=(10,4))
plt.plot(aapl.index, aapl["logreturn"])
plt.title("AAPL Daily Log Returns (2005 - 2025)")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.show()


# In[256]:


r = aapl["logreturn"]

# data checks
checks = {}

checks["n_total"] = r.size
checks["n_null"] = r.isna().sum()
checks["n_posinf"] = np.isposinf(r.fillna(np.inf)).sum()
checks["n_neginf"] = np.isneginf(r.fillna(-np.inf)).sum()

idx = aapl.index
checks["index_is_datetime"] = isinstance(idx, pd.DatetimeIndex)
checks["index_is_monotic_increasing"] = idx.is_monotonic_increasing
checks["n_duplicates_in_index"] = idx.duplicated().sum()

checks["min"] = r.min(skipna=True)
checks["max"] = r.max(skipna=True)

checks
r = aapl["logreturn"].dropna()


# In[257]:


# mean and velocity of aaple

r = aapl["logreturn"].dropna()
mean_daily = r.mean()
std_daily = r.std(ddof=1)

mean_daily, std_daily


# In[258]:


# Sharpe

NUMBER_OF_TRADING_DAYS = 252

ann_mean_log = mean_daily * NUMBER_OF_TRADING_DAYS
ann_vol = std_daily * (NUMBER_OF_TRADING_DAYS ** 0.5)
sharpe = ann_mean_log / ann_vol

ann_simple = np.exp(ann_mean_log) - 1
ann_mean_log, ann_simple, ann_vol, sharpe


# In[259]:


from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt

plot_acf(r, lags=30)
plt.title("ACF of AAPl daily log returns (up to 30 days)")
plt.xlabel("Lag (days)")
plt.ylabel("Autocorrelation")
plt.show()


# We see from this plot that today's return has no meaningful linear relationship with yesterday's, or with any day up to a month back
# First bar is 1 since it's just the correlation of the series with itself
# No momentum, so just because Apple was up yesterday, we can't conclude to buy today

# In[260]:


r2 = r ** 2

plot_acf(r2, lags=30)
plt.title("ACF of Sqaured AAPl daily log returns (up to 30 days)")
plt.xlabel("Lag (days)")
plt.ylabel("Autocorrelation of $r_t^2$")
plt.show()


# The bars are positive and significant, so when a market has a large/small swing today (up or down since it's squared), it's more likely that tomorrow will also have a large/small swing, but we can't predict which direction the swing will be in. 
# 

# In[261]:


import numpy as np

window = 21
roll_vol_ann = r.rolling(window).std(ddof=1) * np.sqrt(NUMBER_OF_TRADING_DAYS)

plt.figure(figsize=(10,4))
plt.plot(roll_vol_ann, linewidth=1)
plt.title(f"AAPL Rolling Annualized Volatility ({window}-day window)")
plt.ylabel("Annualized Volatility")
plt.xlabel("Date")
plt.show()

roll_vol_median = roll_vol_ann.median()
float(roll_vol_median)


# The line is the estimated volatility for each day using only the previous 21 days, without peeking ahead.
# Spikes represent turbulent regions, and troughs are calm regions.
# 
# 25.85% is the median annualized volume across the sample
# 
# This shows when risk was high/low and how quickly it changes.
# 
# 

# In[262]:


# fit t-GARCH(1,1) 
import arch
from arch import arch_model

r_pct = 100 * r
r_pct.head()

am = arch_model(
    r_pct,
    mean="Constant",
    vol="Garch",
    p=1, q=1,
    dist="t"
)

result = am.fit(update_freq=10, disp="off")
print(result.summary())


# We see that a + b is very close to 1, so there is long-lasting volatility

# In[263]:


# get residuals
resid = result.resid
sigma = result.conditional_volatility # pred std dev per day
std_resid = resid / sigma # should be student-t, iid


# In[264]:


#visualize

plt.figure(figsize=(10,4))
plt.plot(resid)
plt.title("Raw residuals (ε_t)")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(sigma)
plt.title("Conditional volatility σ_t")
plt.show()


plt.figure(figsize=(10,4))
plt.plot(std_resid)
plt.title("Standardized residuals (z_t = ε_t/σ_t)")
plt.show()

sns.histplot(std_resid, kde=True)
plt.title("Histogram of standardized residuals")
plt.show()

plot_acf(std_resid, lags=40)
plt.title("ACF of standardized residuals")
plt.show()

plot_acf(std_resid**2, lags=40)
plt.title("ACF of squared standardized residuals")
plt.show()


# In[265]:


# Ljung-Box Test: are the residuals autocorrelated, if past values help predict current ones

lb_resid = acorr_ljungbox(std_resid, lags=[10,20], return_df=True)

lb_sqresid = acorr_ljungbox(std_resid**2, lags=[10,20], return_df=True)

print("Ljunb-Box test on standardized residuals:")
print(lb_resid)
print("\nLB test on squared standardized residuals:")
print(lb_sqresid)


# Any short-term corerlation that exists is small and dies off quickly.
# 
# From the sqaured, we see that they are completley random, since the null hypothesis was no autocorrelation, so no leftover volatility dependence. 

# In[266]:


# Arch-LM Test
# do large shocks still tend to follow each other

arch_test = het_arch(std_resid)
print("ARCH-LM test results:")
print(f"LM stat = {arch_test[0]:.4f}, p-value = {arch_test[1]:.4f}")


# No ARCH effects left, so the volatility model is good, volatility is fully captured.

# In[267]:


# Forecast

forecasts = result.forecast(horizon=100)
forecast_var = forecasts.variance[-1:].T
forecast_var


# This is the model's expected condtiional variance for that next day, given what it knows. They are measures of uncertainity, so how risky the market will be each day
# Right now, the model believes that the volatility is slightly below its long run level, so it is creeping upward

# In[268]:


p = result.params
omega = p['omega']
alpha = p['alpha[1]']
beta  = p['beta[1]']

long_run_var = omega / (1 - alpha - beta)
long_run_vol = long_run_var**0.5
ann_vol = long_run_vol * (252**0.5)

print(f"Long-run variance: {long_run_var:.4f}")
print(f"Long-run daily vol: {long_run_vol:.2f}%")
print(f"Long-run annualized vol: {ann_vol:.2f}%")


# In[269]:


# Visualize future risk levels

plt.figure(figsize=(8,4))
plt.plot(forecast_var**0.5)
plt.title("Forecasted Conditional Variance (Next 10 Days)")
plt.xlabel("Days ahead")
plt.ylabel("Variance")
plt.show()


# 

# In[270]:





mu = float(result.params['mu'])          # mean (%/day)
nu = float(result.params['nu'])          # t degrees of freedom


forecasts = result.forecast(horizon=1)
var_t1 = float(result.forecast(horizon=1).variance.iloc[-1, 0])   # variance for h.01 (percent^2/day)
sigma_t1 = np.sqrt(var_t1)                        # volatility for h.01 (percent/day)

# confidence levels#
levels = {'VaR95': 0.05, 'VaR99': 0.01}


out = {}
for name, alpha in levels.items():
    q = st.t.ppf(alpha, df=nu)                   # negative number
    var_return = mu + sigma_t1 * q               # in percent
    var_loss   = -var_return                      # loss convention (positive)
    out[name] = {'return_VaR_%': var_return, 'loss_VaR_%': var_loss}

out, mu, sigma_t1, nu


# There is a 5% chance the loss exceeds 2.689%

# In[271]:


# Multi day VaR

horizon = 10
alpha = 0.05
forecasts = result.forecast(horizon=horizon)
vars_horizon = forecasts.variance.iloc[-1].values # each days forecast var from model
sigma_h = np.sqrt(np.sum(vars_horizon))  # 10 day forecased volatility

# student t quantile
q = st.t.ppf(alpha,df=nu)

# multi day
var_horizon_return = mu*horizon + sigma_h*q # expected h-day return quantile 
var_horizon_loss = -var_horizon_return


print(f"{horizon}-day 95% VaR (return): {var_horizon_return:.2f}%")
print(f"{horizon}-day 95% VaR (loss):   {var_horizon_loss:.2f}%")


# There is a 7% chance over the next 10 trading thats that the total loss will exceed 7.9%.

# In[272]:


# Expected Shortfall
# If worst a% days happen, how bad are they on average? 
# mean loss beyond VaR


alpha = 0.05 # 94% ES
q = st.t.ppf(alpha, df=nu)
pdf = st.t.pdf(q,df=nu)

multiplier = (nu + q**2) / ((nu - 1) * alpha) * pdf

es_return = mu - sigma_t1 * multiplier
es_loss = -es_return


print(f"1-day 95% ES (return): {es_return:.2f}%")
print(f"1-day 95% ES (loss):   {es_loss:.2f}%")


# We see that if tomorrow is included in the worst 5% of days, the average loss is about 1.19%

# In[273]:


# Test 1-day Var to check if model matches reality
# Do x% of days break the 95%VaR and do they happen iid


alpha = 0.05
history = 1000
rets = r_pct.loc['2020-01-01':].dropna()
dates = rets.index
returns = rets.values

VaR_pred = []
actual = []

for t in range(history, len(returns)-1):
    #fit the model up to day t
    y = pd.Series(rets[t-history:t], index=dates[t-history:t])
    am = arch_model(y, mean="Constant", vol="Garch", p=1, q=1, dist="t")
    fitted = am.fit(disp="off")

    # params and forecast next-day variance
    mu = float(fitted.params['mu'])
    nu = float(fitted.params['nu'])
    var_next = float(fitted.forecast(horizon=1).variance.iloc[-1,0])
    sigma_next = np.sqrt(var_next)

    # computer 1-dat VaR
    q = st.t.ppf(alpha, df=nu)
    VaR_next = mu + sigma_next*q
    VaR_pred.append(VaR_next)

    actual.append(returns[t+1])

backtest = pd.DataFrame({
    "Date": dates[history+1:],
    "Return": actual,
    "VaR": VaR_pred
}).set_index("Date")

# check when they exceed

backtest["Exceed"] = (backtest["Return"] < backtest["VaR"]).astype(int)
backtest.head()



# In[274]:


plt.figure(figsize=(10,5))
plt.plot(backtest["Return"], label="Actual Return", color="gray")
plt.plot(backtest["VaR"], label="Predicted 95% VaR", color="red")
plt.fill_between(backtest.index, backtest["VaR"], min(backtest["Return"]), color="red", alpha=0.1)
plt.scatter(backtest.index[backtest["Exceed"]==1], backtest["Return"][backtest["Exceed"]==1],
            color="black", marker="x", label="Exceedance")
plt.legend(); plt.title("1-Day 95% VaR Backtest"); plt.show()


# In[275]:


# Frequency (Kupiec Test)

n = len(backtest)
x = backtest["Exceed"].sum()
p_hat = x / n
LR_uc = -2 * ((n - x) * np.log((1 - alpha) / (1 - p_hat)) + x * np.log(alpha / p_hat) )
p_uc = 1 - st.chi2.cdf(LR_uc, df=1)


print(f"Exceedances {x}/{n} = {p_hat:.2%}, Kupiec p-value = {p_uc:.3f}")


# From here we see that the model overestiamtes risk.

# In[276]:


import numpy as np, pandas as pd, scipy.stats as st
from arch import arch_model

alpha = 0.05
window = 750                 # ~3y; use 500 for faster response
series = r_pct.loc['2020-01-01':].dropna()
dates, rets = series.index, series.values

VaR, realized = [], []
for t in range(window, len(rets)-1):
    y = pd.Series(rets[t-window:t], index=dates[t-window:t])
    am = arch_model(y, mean="Constant", vol="Garch", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    mu, nu = float(res.params['mu']), float(res.params['nu'])
    sig1 = float(res.forecast(horizon=1).variance.iloc[-1,0])**0.5
    q = st.t.ppf(alpha, df=nu)
    VaR.append(mu + sig1*q)           # return VaR (%)
    realized.append(rets[t+1])

bt = pd.DataFrame({"Return": realized, "VaR": VaR}, index=dates[window+1:])
bt["Exceed"] = (bt["Return"] <= bt["VaR"]).astype(int)

# Kupiec
n, x = len(bt), int(bt["Exceed"].sum())
p_hat = x/n
LR_uc = -2*np.log(((1-alpha)**(n-x)*alpha**x)/((1-p_hat)**(n-x)*p_hat**x))
p_uc = 1 - st.chi2.cdf(LR_uc, 1)
print(f"POST-2020 95% VaR: breaches {x}/{n} = {p_hat:.2%}, Kupiec p={p_uc:.3f}")


# Modeling
# 

# In[277]:


# Target: tomorrow's direction at t+1

df = aapl[['Adj Close', 'logreturn']].copy()

df['y'] = (df['logreturn'].shift(-1) > 0).astype(int)

df = df.dropna(subset=['logreturn', 'y'])
df


# In[278]:


#integrity checks

df[['logreturn', 'y']].isna().sum(), df['y'].mean(),


# In[279]:


# Feature Set v1 to predict t+1


# Momentum

# 5 day momentum
df['mom5'] = df['logreturn'].rolling(5).mean()

# 10 day momentum
df['mom10'] = df['logreturn'].rolling(10).mean()

# rate of change for 5 days
df['roc5'] = df['Adj Close'].pct_change(5)
df, df[['mom5','mom10','roc5']].corr(), 


# In[280]:


# Volatlity

# 10 day volatility
df['vol10'] = df['logreturn'].rolling(10).std()
df['vol20'] = df['logreturn'].rolling(20).std()

df


# In[281]:


# Trend & Deviation

# z-score vs 20 day mean
# close - MA / SD

ma20 = df['Adj Close'].rolling(20).mean()
sd20 = df['Adj Close'].rolling(20).std()

df['z_ma20'] = (df['Adj Close'] - ma20) / sd20


# Bollinger %b: scale relative to range
roll_min = df['Adj Close'].rolling(20).min()
roll_max = df['Adj Close'].rolling(20).max()
df['bbp20'] = (df['Adj Close'] - roll_min) / (roll_max - roll_min)

df[['z_ma20','bbp20']].describe(),df['bbp20'].between(0,1).mean()  


# In[282]:


# Skewness and Kurtosis

df['skew20'] = df['logreturn'].rolling(20).skew()
df['kurt20'] = df['logreturn'].rolling(20).kurt()
df


# In[283]:


# Clean up Nans

feature_cols = [
    'mom5', 'mom10', 'roc5',
    'vol10', 'vol20',
    'z_ma20', 'bbp20',
    'skew20', 'kurt20'
]

df_no_nan = df[['logreturn', 'y'] + feature_cols].dropna()
df_no_nan.shape, df_no_nan.isna().sum(), df_no_nan[feature_cols].describe().T.head(10)
df_no_nan.loc['2010']


# In[284]:


# Forward split function

feature_cols = [
    'mom5', 'mom10', 'roc5',
    'vol10', 'vol20',
    'z_ma20', 'bbp20',
    'skew20', 'kurt20'
]

def forward_split(df, test_year, train_start="2005-02-01", embargo_days=5, feature_cols=feature_cols):


    test_mask = (df.index.year == test_year)
    test_idx = df.index[test_mask]
    test_start = test_idx[0]
    test_end = test_idx[-1]

    test_start_pos = df.index.get_loc(test_start)
    embargo_pos = max(0, test_start_pos - embargo_days)
    embargo_start = df.index[embargo_pos]

    train_start_dt = pd.to_datetime(train_start)

    train_mask = (df.index >= train_start_dt) & (df.index < embargo_start)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['y']


    X_test = test_df[feature_cols]
    y_test = test_df['y']

    meta = {
        "test_year": test_year,
        "embargo_days": embargo_days,
        "date_train_start": train_df.index.min(),
        "date_train_end": train_df.index.max(),
        "date_test_start": test_start,
        "date_test_end": test_end,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "class_balance_train": float(y_train.mean()),
        "class_balance_test": float(y_test.mean()),
    }

    return (X_train, y_train, X_test, y_test, meta)

forward_split(df_no_nan, 2020)


    

    


# In[285]:


# logistic regression for 2020

feature_cols = [
    'mom5','mom10',
    'vol10','vol20',
    'z_ma20','bbp20',
    'skew20','kurt20'
]

X_train, y_train, X_test, y_test, meta = forward_split(
    df_no_nan,
    test_year=2020
)

meta


# In[286]:


# no missing values inside either split
assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0, "NaNs in features"
assert y_train.isna().sum() == 0 and y_test.isna().sum() == 0, "NaNs in target"

# identical column order in train and test
assert X_train.columns.tolist() == X_test.columns.tolist(), "Train/Test feature columns misaligned"


# In[287]:


# Fit Scaler on train only

scaler = StandardScaler()

scaler.fit(X_train)

Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)


# In[288]:


# Train logistic regression

logit = LogisticRegression(
    C = 0.2,
    solver = "liblinear",
    max_iter=1000
)

# Fit on train only
logit.fit(Xtr, y_train)

# predict on test
proba_te = logit.predict_proba(Xte)[:,1]
pred_te = (proba_te >= 0.5).astype(int)


# coefficients
coefs = pd.Series(logit.coef_.ravel(), index=X_train.columns).sort_values()
coefs


# In[289]:


# Check metrics

acc = accuracy_score(y_test, pred_te)
auc = roc_auc_score(y_test, proba_te)
brier = brier_score_loss(y_test, proba_te)

acc, auc, brier


# In[290]:


# Walk forward test for multiple years

years = [2020, 2021, 2022, 2023, 2024]
results = []
for year in years:
    X_train, y_train, X_test, y_test, meta = forward_split(df_no_nan, test_year=year)

    scaler.fit(X_train)

    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    logit.fit(Xtr, y_train)

    proba_te = logit.predict_proba(Xte)[:,1]
    pred_te  = (proba_te >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_te)
    auc = roc_auc_score(y_test, proba_te)
    brier = brier_score_loss(y_test, proba_te)

    results.append({
        "year": year,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "acc": acc,
        "auc": auc,
        "brier": brier,
        "always_up":  y_test.mean()
    })


results_logistic = pd.DataFrame(results)
results_logistic.to_csv("../results/results_logistic.csv", encoding='utf-8', index=False, header=True)
results_logistic


# In[ ]:





# In[308]:


# XGBoost for one year
feature_cols_xgb = ['mom5', 'mom10', 'vol10', 'vol20', 'z_ma20', 'bbp20']

X_train, y_train, X_test, y_test, meta = forward_split(df_no_nan, test_year=year, feature_cols=feature_cols_xgb)
validation_tail_days = 180
X_tr = X_train.iloc[:-validation_tail_days, :]
y_tr = y_train.iloc[:-validation_tail_days]
X_val = X_train.iloc[-validation_tail_days:, :]
y_val = y_train.iloc[-validation_tail_days:]


xgb = XGBClassifier(
    n_estimators=200,
    max_depth=2,
    learning_rate = 0.01,
    min_child_weight=10,
    gamma=1.0,
    reg_lambda=5.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric='logloss',
    n_jobs=1,
    scale_pos_weight=1.0
)



xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], )

hist = xgb.evals_result_
val_loss = np.array(hist['validation_0']['logloss'])
best_round = int(val_loss.argmin()) + 1


val_proba = xgb.predict_proba(X_val)[:,1]
thresholds = np.linspace(0.45, 0.6, 31)
best_th = max(thresholds, key=lambda t: accuracy_score(y_val, (val_proba >= t).astype(int)))


# In[ ]:


xgb_best = XGBClassifier(
    n_estimators=best_round,
    max_depth=3,
    learning_rate = 0.03,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=1
)

xgb_best.fit(X_train, y_train)

proba_te = xgb_best.predict_proba(X_test)[:, 1]
pred_te_05 = (proba_te >= 0.5).astype(int)
pred_te_tuned = (proba_te >= best_th).astype(int)



acc   = accuracy_score(y_test, pred_te_05)
auc = roc_auc_score(y_test, proba_te)  
brier = brier_score_loss(y_test, proba_te)

print(f"Accuracy={acc:.3f}, AUC={auc:.3f}, Brier={brier:.3f}")


# In[307]:


# xgb DF
# Walk-forward XGB (like logistic)
xgb_results = []
for year in years:
    X_train, y_train, X_test, y_test, meta = forward_split(df_no_nan, test_year=year, feature_cols=feature_cols)
    
    # Split train into train/val for threshold tuning
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]
    
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=1.0,  # adjust if imbalanced
        random_state=42
    )
    xgb_model.fit(X_tr, y_tr)
    
    proba_te = xgb_model.predict_proba(X_test)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, pred_te)
    auc = roc_auc_score(y_test, proba_te)
    brier = brier_score_loss(y_test, proba_te)
    
    xgb_results.append({"year": year, "acc": acc, "auc": auc, "brier": brier})

results_xgb = pd.DataFrame(xgb_results)
results_xgb.to_csv("../results/results_xgb.csv", encoding='utf-8', index=False, header=True)
results_xgb


# In[294]:


import os, sys, importlib
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm_api import nl_api


# In[295]:


importlib.reload(nl_api)
with open(os.path.join(project_root, "llm_api", "catalog.json")) as f:
    catalog=json.load(f)

nl_api.set_env(
    df_ml=df_no_nan,
    results_logistic=results_logistic,
    results_xgb=results_xgb,
    catalog=catalog
)


nl_api.get_metric(2024, "logistic", "ACC")
nl_api.list_years("logistic")
nl_api.explain("bbp20")
nl_api.plot_feature("vol10")
nl_api.plot_scatter("mom10", "vol10")
nl_api.compare_models("acc")



