import pandas as pd
import matplotlib.pyplot as plt

CATALOG  = None

ENV = {}

def set_env(*, df_ml=None, results_logistic=None, results_xgb=None, catalog=None):
    global ENV, CATALOG
    CATALOG = catalog
    ENV = {
        "df_ml": df_ml,
        "results_logistic": results_logistic,
        "results_xgb": results_xgb,
        "catalog": catalog
    }


def get_metric(year, model, metric):
    # Returns the metric needed, given the year and model
    
    # check constraints
    year = int(year)
    model = model.lower()
    metric = metric.lower()

    if model not in CATALOG['models']:
        print(f"{model} not valid")
        return
    if metric not in CATALOG['metrics']:
        print(f"{metric} not valid")
        return
    
    
    df_map = {
        "logistic": ENV.get("results_logistic"),
        "xgb": ENV.get("results_xgb")
    }

    df = df_map.get(model)
    if df is None:
        print(f"No results dataframe loaded for model '{model}'. Did you call set_env?")
        return

    colmap = {c.lower(): c for c in df.columns}
    if "year" not in colmap:
        print("Results DataFrame must have 'year' column")
        return
    if metric not in colmap:
        print(f"Results DataFrame missing metric column '{metric}'. Have: {list(df.columns)}")
        return

    year_col = colmap["year"]
    row = df[df[year_col] == year]
    if row.empty:
        print(f"{year} not found")
        return

    value = row.iloc[0][colmap[metric]]
    if value is None or (hasattr(value, "isna") and pd.isna(value)):
        print(f"Metric '{metric}' is NaN for year={year}, model='{model}")
        return
    return float(value) 




def list_years(model):
    # lists all years available for a given model
    model = model.lower()
    if model not in CATALOG["models"]:
        print(f"{model} not valid")
        return

    df_map = {
        "logistic": ENV.get("results_logistic"),
        "xgb": ENV.get("results_xgb")
    }

    df = df_map.get(model)
    if df is None:
        print(f"No results dataframe loaded for model '{model}'. Did you call set_env?")
        return

    years = df["year"].tolist()

    return years
    



def plot_feature(feature, start_date=None, end_date=None):
    # Visualize how a chosen feature evolves over time
    allowed_features = CATALOG['allowed_features']
    print(allowed_features)
    df = ENV['df_ml']
    feat = str(feature).lower()

    if feat not in allowed_features:
        raise ValueError(f"{feature} not allowed. Allowed: {allowed}")


    feature_norm = str(feature).lower()
    allowed = [f.lower() for f in CATALOG.get("allowed_features", [])]
    if feature_norm not in allowed:
        raise ValueError(f"{feature} not allowed. ALlowed: {allowed}")
    

    # df = df.reset_index().rename(columns={df.columns[0]: "Date"})


    if start_date or end_date:
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = df.index.min()
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = df.index.max()
        d = df.loc[start:end]
    else:
        d = df


    plt.figure(figsize=(10,4))
    plt.plot(d.index, d[feature], label=feature, linewidth=1.5)
    plt.title(f"{feature} over time")
    plt.ylabel(f"{feature} value")
    plt.xlabel("Date")
    plt.show()

    return

def plot_scatter(x_feature, y_feature, start_date=None, end_date=None):
    # shows the realationship between two features in the most recent N days
    df = ENV["df_ml"]
    allowed = [f.lower() for f in CATALOG.get("allowed_features", [])]

    xf, yf = x_feature.lower(), y_feature.lower()
    for f in (xf, yf):
        if f not in allowed:
            raise ValueError(f"{f} not allowed. ALlowed: {allowed}")
    
    x_col = next((c for c in df.columns if c.lower() == xf), None)
    y_col = next((c for c in df.columns if c.lower() == yf), None)

    if start_date or end_date:
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = df.index.min()
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = df.index.max()
        d = df.loc[start:end]
    else:
        d = df
    
    plt.figure(figsize=(10,4))
    plt.scatter(d[x_col], d[y_col], s=10, alpha=0.6)
    plt.ylabel(y_col)
    plt.xlabel(x_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.show()

    return
    

def explain(term):
    # returns the text explanation for the term from the glossary
    glossary = CATALOG["glossary"]
    term = term.lower()
    if term in glossary:
        return glossary[term]
    
    aliases = {
        "bbp20": "bollinger_band_percent_b",
        "z_ma20": "z_score_ma20",
        "logreturn": "log_return",
        "adj_close": "adjusted_close"
    }

    if term in aliases and aliases[term] in glossary:
        return glossary[aliases[term]]

    raise ValueError(
        f"No explanation found for '{term}'."
        f"Try one of {list(glossary.keys())}"

    )


def compare_models(metric):
    metric = metric.lower()
    if metric not in [m.lower() for m in CATALOG['metrics']]:
        print(f"{metric} not valid. Allowed: {CATALOG['metrics']}")
        return
    
    df_log = ENV.get("results_logistic")
    df_xgb = ENV.get("results_xgb")

    if df_log is None or df_xgb is None:
        print("Missing results DataFrames - did you call set_env()?")
    
    colmap_log = {c.lower(): c for c in df_log.columns}
    colmap_xgb = {c.lower(): c for c in df_xgb.columns}

    if "year" not in colmap_log or metric not in colmap_log:
        print(f"Logistic missing columns: {df_log.columns}")
        return
    if "year" not in colmap_xgb or metric not in colmap_xgb:
        print(f"XGB missing columns: {df_xgb.columns}")
        return
    
    merged = pd.merge(
        df_log[[colmap_log["year"], colmap_log[metric]]],
        df_xgb[[colmap_xgb["year"], colmap_xgb[metric]]],
        on=colmap_log["year"],
        suffixes=("_logistic", "_xgb")
    ).rename(columns={colmap_log["year"]: "year"})
    
    plt.figure(figsize=(8,4))
    plt.plot(merged["year"], merged[f"{metric}_logistic"], label="Logistic", marker="o")
    plt.plot(merged["year"], merged[f"{metric}_xgb"], label="XGB", marker="o")
    plt.title(f"{metric.upper()} Comparison: Logisitc vs XGB")
    plt.xlabel("Year")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.show()

    return merged
