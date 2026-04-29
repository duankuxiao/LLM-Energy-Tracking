import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# ==========================================
# 1. Anchored exponential decay fitting
# ==========================================
def fit_anchored_exponential_decay(x, y):
    """
    Fit an anchored exponential decay model.
    The forecast starts from the last observed value to keep the projection continuous.
    """

    # Fit y = a * exp(-b * t) + c.
    def exp_func(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Normalize the time axis so the first historical year starts at 0.
    x_offset = x[0]
    x_norm = x - x_offset

    # Initial parameter guess and bounds.
    y_min, y_max = np.min(y), np.max(y)
    p0 = [y_max - y_min, 0.1, y_min]

    # Constrain the model to non-negative values and a non-negative decay rate.
    try:
        popt, pcov = curve_fit(exp_func, x_norm, y, p0=p0,
                               bounds=([0, 1e-5, 0], [np.inf, 1.0, np.inf]),
                               maxfev=10000)
    except RuntimeError:
        # Fall back to a conservative 2% annual decay if fitting fails.
        popt = [y[-1], 0.02, 0]

    fit_a, fit_b, fit_c = popt

    # Use the last historical observation as the forecast anchor.
    last_year = x[-1]
    last_val = y[-1]

    # Calculate R2 as a historical fit quality reference.
    residuals = y - exp_func(x_norm, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Adjust the fitted floor if it would otherwise exceed the anchor value.
    effective_c = fit_c if fit_c < last_val else last_val * 0.5

    # Forecast from the anchor year using the fitted decay rate.
    def model_func(future_years):
        future_years = np.atleast_1d(future_years)
        delta_t = future_years - last_year
        preds = (last_val - effective_c) * np.exp(-fit_b * delta_t) + effective_c
        return preds

    return model_func, r2, f"Exp-Anchor(b={fit_b:.3f})"


# ==========================================
# 2. Main processing function for wide-format input data
# ==========================================
def Water_factor_regression(file_path, output_path):
    # Load source data.
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: file not found.")
        return

    # Detect country columns from a wide table with Year as the time column.
    year_col = 'Year'
    if year_col not in df.columns:
        print("Error: 'Year' column not found.")
        return

    target_countries = [c for c in df.columns if c != year_col]

    # Convert the wide table into long format for per-country fitting.
    df_long = df.melt(id_vars=[year_col], value_vars=target_countries,
                      var_name='Country', value_name='Water_Factor')

    all_countries_data = []
    all_historical_data = []

    print(f"{'Country':<20} | {'R2':<8} | {'Fit note'}")
    print("-" * 55)

    for country in target_countries:
        country_df = df_long[df_long['Country'] == country].copy()

        # Clean and sort historical records.
        country_df = country_df.dropna(subset=[year_col, 'Water_Factor'])
        country_df[year_col] = country_df[year_col].astype(int)
        country_df = country_df.sort_values(year_col)

        if country_df.empty:
            continue

        # Keep historical data for the combined output.
        all_historical_data.append(country_df)

        x = country_df[year_col].values
        y = country_df['Water_Factor'].values

        # Fit the anchored decay model.
        fit_model, r2, fit_method = fit_anchored_exponential_decay(x, y)
        print(f"{country:<20} | {r2:.4f}   | {fit_method}")

        # Generate projected water factors for 2019-2030.
        future_years = np.arange(2019, 2031)

        predicted_values = fit_model(future_years)
        predicted_values = np.maximum(predicted_values, 0)

        # Build the prediction result table.
        temp_df = pd.DataFrame({
            'Country': country,
            'Year': future_years,
            'Water_Factor': np.round(predicted_values, 4)
        })

        all_countries_data.append(temp_df)

    # Save the combined history and prediction output.
    if all_countries_data:
        prediction_df = pd.concat(all_countries_data, ignore_index=True)

        history_df = pd.concat(all_historical_data, ignore_index=True)
        full_df = pd.concat([history_df, prediction_df], ignore_index=True)

        # Remove duplicated country-year rows.
        full_df = full_df.drop_duplicates(subset=['Country', 'Year'], keep='first')

        # Export a wide table with Year as rows and countries as columns.
        # index=Year, columns=Country
        full_wide_df = full_df.pivot(index='Year', columns='Country', values='Water_Factor')
        full_wide_df = full_wide_df.sort_index().sort_index(axis=1)

        full_wide_df.to_csv(output_path)

        print("-" * 55)
        print(f"Processing completed. Full data (2010-2030) saved to: {output_path}")
    else:
        print("No data generated.")



if __name__ == '__main__':
    # ==========================================
    # Example run
    # ==========================================
    Water_factor_regression('../dataset/Grid_water_factors_2010_2018.csv', '../results/Grid_water_factors_2010_2030.csv')


