import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data_path = 'data_cleaned.csv'  
df = pd.read_csv(data_path)

print("Columns:", df.columns)
print(df.head())


emission_col = 'CO2_emissions'

required_cols = ['Country', 'Year', emission_col]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in data.")

df = df[required_cols].dropna(subset=[emission_col])
df = df.sort_values(['Country', 'Year'])

def create_lag_features(group, lags=[1, 2, 3]):
    for lag in lags:
        group[f'lag_{lag}'] = group[emission_col].shift(lag)
    return group

df = df.groupby('Country').apply(create_lag_features)
df = df.dropna()

# --- Step 4: Prepare features and target ---
feature_cols = ['Year', 'lag_1', 'lag_2', 'lag_3']

# Encode Country as categorical integer
df['Country_enc'] = df['Country'].astype('category').cat.codes
feature_cols.append('Country_enc')

X = df[feature_cols]
y = df[emission_col]


train = df[df['Year'] <= 2010]
test = df[df['Year'] > 2010]

X_train = train[feature_cols]
y_train = train[emission_col]

X_test = test[feature_cols]
y_test = test[emission_col]


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


plt.figure(figsize=(10,6))
plt.plot(test['Year'], y_test, label='True CO2 emissions', marker='o')
plt.plot(test['Year'], y_pred, label='Predicted CO2 emissions', marker='x')
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.title('True vs Predicted CO2 Emissions (Test set)')
plt.legend()
plt.show()


def forecast_next_years(df, model, country_name, years_to_forecast=5):
    country_data = df[df['Country'] == country_name].sort_values('Year').copy()
    last_year = country_data['Year'].max()
    
    for i in range(1, years_to_forecast + 1):
        next_year = last_year + i
    
        try:
            lag_1 = country_data.loc[country_data['Year'] == next_year - 1, emission_col].values[0]
            lag_2 = country_data.loc[country_data['Year'] == next_year - 2, emission_col].values[0]
            lag_3 = country_data.loc[country_data['Year'] == next_year - 3, emission_col].values[0]
        except IndexError:
            print(f"Not enough data to create lag features for year {next_year}. Stopping forecast.")
            break
        
        country_enc = country_data['Country_enc'].iloc[0]
        
        X_pred = pd.DataFrame({
            'Year': [next_year],
            'lag_1': [lag_1],
            'lag_2': [lag_2],
            'lag_3': [lag_3],
            'Country_enc': [country_enc]
        })
        
        pred_emission = model.predict(X_pred)[0]
        
        new_row = pd.DataFrame({
            'Country': [country_name],
            'Year': [next_year],
            emission_col: [pred_emission],
            'lag_1': [np.nan], 'lag_2': [np.nan], 'lag_3': [np.nan], 'Country_enc': [country_enc]
        })
        
        country_data = pd.concat([country_data, new_row], ignore_index=True)
        
        country_data.loc[country_data.index[-1], 'lag_1'] = country_data.loc[country_data.index[-2], emission_col]
        country_data.loc[country_data.index[-1], 'lag_2'] = country_data.loc[country_data.index[-3], emission_col]
        country_data.loc[country_data.index[-1], 'lag_3'] = country_data.loc[country_data.index[-4], emission_col]
    
    return country_data.tail(years_to_forecast)[['Year', emission_col]]

forecasted = forecast_next_years(df, model, country_name='United States', years_to_forecast=5)
print("Forecasted CO2 emissions for next 5 years:")
print(forecasted)
