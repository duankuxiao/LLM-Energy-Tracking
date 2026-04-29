import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


R2_TARGET = 0.9


def safe_r2_score(y_true, y_pred):
    if len(y_true) < 2:
        return 1.0
    return r2_score(y_true, y_pred)


def fit_poly_model(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coefficients)
    y_pred = poly_func(x)
    return poly_func, safe_r2_score(y, y_pred), f"poly_deg_{degree}"


def fit_exp_model(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("y must be positive for exp fit")
    coefficients = np.polyfit(x, np.log(y), 1)
    a, b = coefficients

    def model(x_new):
        x_new = np.asarray(x_new, dtype=float)
        return np.exp(a * x_new + b)

    y_pred = model(x)
    return model, safe_r2_score(y, y_pred), "exp"


def select_fit_model(x, y, r2_target=R2_TARGET):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    attempts = []
    if len(x) >= 3:
        attempts.append(lambda: fit_poly_model(x, y, 2))
    if len(x) >= 4:
        attempts.append(lambda: fit_poly_model(x, y, 3))
    if len(x) >= 2:
        attempts.append(lambda: fit_poly_model(x, y, 1))
    if len(x) >= 2:
        attempts.append(lambda: fit_exp_model(x, y))

    best = None
    for attempt in attempts:
        try:
            result = attempt()
        except Exception:
            continue
        model, r2, method = result
        if best is None or r2 > best[1]:
            best = result
        if r2 >= r2_target:
            return result

    return best


target_countries = [
    "United States of America", "China", "Japan", "France", "India", "Singapore",
    "Canada", "Germany", "United Kingdom", "Australia", "Italy", "South Korea",
    "South Africa", "Ireland", "United Arab Emirates", "Brazil", "Israel",
    "Netherlands", "Spain", "Sweden", "Belgium", "Norway",
    "Poland", "Switzerland"
]


def fit_anchored_exponential_decay(x, y):
    """
    Fit an anchored exponential decay model.
    The forecast starts from the last observed value to avoid a first-year jump.
    """

    # Fit y = a * exp(-b * t) + c to estimate the decay rate and floor.
    # y = a * exp(-b * (t - t0)) + c
    def exp_func(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Normalize the time axis so the first historical year starts at 0.
    x_offset = x[0]
    x_norm = x - x_offset

    # Initial parameter guess.
    y_min = np.min(y)
    y_max = np.max(y)
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

    def model_func(future_years):
        future_years = np.atleast_1d(future_years)
        # Forecast from the anchor year using the fitted decay rate.
        delta_t = future_years - last_year
        preds = (last_val - effective_c) * np.exp(-fit_b * delta_t) + effective_c
        return preds

    return model_func, r2, f"Exp-Anchor(b={fit_b:.3f})"


def Carbon_emission_factor_regression(file_path, output_path, model_type='exponential'):
    """
    model_type: 'linear' uses automatic model selection; 'exponential' uses anchored decay.
    """
    # Load source data.
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: file not found.")
        return

    df_filtered = df[df['Area'].isin(target_countries)].copy()
    all_countries_data = []
    all_historical_data = []

    print(f"{'Country':<15} | {'Model':<12} | {'R2':<8} | {'Note'}")
    print("-" * 55)

    for country in target_countries:
        country_df = df_filtered[df_filtered['Area'] == country].copy()

        if country_df.empty:
            continue

        # Clean and sort historical records.
        country_df['Year'] = pd.to_numeric(country_df['Year'], errors='coerce')
        country_df = country_df.dropna(subset=['Year', 'Grid_Carbon_Factor_gco2_per_kwh'])
        country_df['Year'] = country_df['Year'].astype(int)

        country_df = country_df.sort_values('Year')

        # Keep historical data for the combined output.
        hist_temp = country_df[['Area', 'Year', 'Grid_Carbon_Factor_gco2_per_kwh']].copy()
        hist_temp.columns = ['Country', 'Year', 'Grid_Carbon_Factor_gco2_per_kwh']
        all_historical_data.append(hist_temp)

        x = country_df['Year'].values
        y = country_df['Grid_Carbon_Factor_gco2_per_kwh'].values

        # Select and fit the forecast model.
        fit_model = None
        r2 = None
        fit_method = ""

        if model_type == 'exponential':
            fit_model, r2, fit_method = fit_anchored_exponential_decay(x, y)
            if fit_model is None:
                print(f"{country:<15} | exponential fit failed, trying linear fallback...")
                fit_model, r2, fit_method = select_fit_model(x, y)
        else:
            fit_model, r2, fit_method = select_fit_model(x, y)

        if fit_model is None or r2 is None:
            print(f"{country:<15} | fit failed   | N/A      | skipped")
            continue

        trend_note = f"{fit_method}"
        print(f"{country:<15} | {fit_method:<12} | {r2:.4f}   | {trend_note}")

        # Generate projected carbon factors for 2025-2030.
        future_years = np.arange(2025, 2031)
        predicted_values = fit_model(future_years)

        # Enforce a non-negative physical lower bound.
        predicted_values = np.maximum(predicted_values, 0)

        temp_df = pd.DataFrame({
            'Country': country,
            'Year': future_years,
            'Grid_Carbon_Factor_gco2_per_kwh': np.round(predicted_values, 2)
        })

        all_countries_data.append(temp_df)

    # Save prediction and combined history outputs.
    if all_countries_data:
        # Long Format
        final_df = pd.concat(all_countries_data, ignore_index=True)
        final_df.to_csv(output_path, index=False)

        # Wide Format (Prediction Only)
        if output_path.lower().endswith('.csv'):
            base_path = output_path[:-4]
        else:
            base_path = output_path

        wide_output_path = base_path + '_by_year.csv'
        wide_df = final_df.pivot(index='Year', columns='Country', values='Grid_Carbon_Factor_gco2_per_kwh')
        wide_df = wide_df.sort_index().sort_index(axis=1)
        wide_df.to_csv(wide_output_path)

        # Full History + Prediction Wide Format
        full_output_path = base_path + '_full_history_and_prediction.csv'
        if all_historical_data:
            history_df = pd.concat(all_historical_data, ignore_index=True)
            full_combined_df = pd.concat([history_df, final_df], ignore_index=True)
            full_combined_df = full_combined_df.drop_duplicates(subset=['Country', 'Year'], keep='first')

            full_wide_df = full_combined_df.pivot(index='Year', columns='Country', values='Grid_Carbon_Factor_gco2_per_kwh')
            full_wide_df = full_wide_df.sort_index().sort_index(axis=1)
            full_wide_df.to_csv(full_output_path)

            print("-" * 55)
            print(f"Processing completed. Mode: {model_type}")
            print(f"Full data saved to: {full_output_path}")
    else:
        print("No data generated.")


if __name__ == '__main__':
    # Example run. Replace the input path with the actual source file if needed.
    Carbon_emission_factor_regression('../dataset/Carbon_emission_factors_2010_2018.csv', '../results/Carbon_emission_factors_2025-2030.csv')
