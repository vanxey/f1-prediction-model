import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def prepare_and_predict(df):
    required_columns = [
        'driver_number', 'lap_number', 'pit_duration', 'position', 'wind_speed',
        'track_temperature', 'air_temperature', 'humidity', 'lap_start', 'lap_end',
        'stint_number', 'tyre_age_at_start', 'compound', 'session_key', 'team_name', 
        'full_name', 'session_type', 'rainfall'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("Missing columns:", missing)
        return df

    numeric_columns = [
        'driver_number', 'lap_number', 'pit_duration', 'position', 'wind_speed',
        'track_temperature', 'air_temperature', 'humidity', 'lap_start',
        'lap_end', 'stint_number', 'tyre_age_at_start', 'rainfall'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['compound'] = df['compound'].astype('category').cat.codes
    df['tyre_age_at_start'] = df['tyre_age_at_start'].fillna(df['tyre_age_at_start'].median())
    df['session_type'] = df['session_type'].astype(str).str.strip().str.lower()
    race_df = df[df['session_type'] == 'race'].copy()
    
    race_df['lap_number'] = race_df['lap_number'].astype(int)
    race_df['max_lap'] = race_df.groupby(['session_key', 'driver_number'])['lap_number'].transform('max')
    df_final = race_df[race_df['lap_number'] == race_df['max_lap']].drop_duplicates(['session_key', 'driver_number'])

    features = [
        'compound', 'wind_speed', 'track_temperature', 
        'air_temperature', 'humidity', 'pit_duration', 
        'rainfall', 'lap_start', 'lap_end',
        'stint_number', 'tyre_age_at_start'
    ]
    df_final = df_final.dropna(subset=features + ['position'])

    X = df_final[features]
    y = df_final['position']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    predicted = model.predict(X_scaled)

    df_final['predicted_score'] = predicted
    df_final['predicted_position'] = df_final.groupby('session_key')['predicted_score'].rank(method='first', ascending=True).astype(int)

    print(df_final[['session_key', 'session_type', 'team_name', 'driver_number',
                    'full_name', 'position', 'predicted_position']]
          .sort_values(['session_key', 'predicted_position']).head(20))

    print("\nClassification Report:")
    print(classification_report(y, predicted))

    return df_final

df = pd.read_csv('merged_openf1_data_9.csv', delimiter=',', low_memory=False, dtype={'compound': 'string'})
df.columns = df.columns.str.strip().str.lower()
df_result = prepare_and_predict(df)
