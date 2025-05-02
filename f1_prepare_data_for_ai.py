import dask.dataframe as dd
import pyarrow

def merge_openf1_data():
    stints = dd.read_csv("csv/stints.csv", delimiter=";", dtype={"track_temperature": "object", "tyre_age_at_start": "object"})
    weather = dd.read_csv("csv/weather.csv", delimiter=";")
    sessions = dd.read_csv("csv/sessions.csv", delimiter=";")
    meetings = dd.read_csv("csv/meetings.csv", delimiter=";")
    final_pos = dd.read_csv("csv/final_pos.csv", delimiter=";")
    drivers = dd.read_csv("csv/drivers.csv", delimiter=";")
    pits = dd.read_csv("csv/pits.csv", delimiter=";")

    def removeMeetingKeyDuplicates(df):
        df = df.rename(columns={"meeting_key_x": "meeting_key"})
        df = df.drop(columns=["meeting_key_y"], errors="ignore")
        return df

    df = sessions.merge(meetings, on="meeting_key", how="left")
    df = removeMeetingKeyDuplicates(df)

    weather = weather.drop_duplicates(subset='session_key')
    df = df.merge(weather, on="session_key", how="left")
    df = removeMeetingKeyDuplicates(df)
    
    df = df.merge(final_pos, on=["session_key", "meeting_key"], how="left")
    df = removeMeetingKeyDuplicates(df)
    
    df = df.merge(drivers, on=["driver_number", "session_key"], how="left")
    df = removeMeetingKeyDuplicates(df)

    df = df.merge(pits, on=["driver_number", "session_key"], how="left")
    df = removeMeetingKeyDuplicates(df)

    df = df.merge(stints, on=["driver_number", "session_key"], how="left")
    df = removeMeetingKeyDuplicates(df)

    non_empty_columns = [col for col in df.columns if df[col].isnull().sum().compute() < len(df)]
    df = df[non_empty_columns]

    df['pit_duration'] = df['pit_duration'].fillna(0)
    df['lap_number'] = df['lap_number'].fillna(df['lap_number'].mean())

    df = df.drop(columns=[
        "Unnamed: 12",
        "circuit_key_y",
        "circuit_short_name_y",
        "country_code_y",
        "country_key_y",
        "country_name_y",
        "date_start_y",
        "gmt_offset_y",
        "location_y",
        "year_y",
        "date_y"
    ], errors="ignore")

    df = df.rename(columns={
        "circuit_key_x": "circuit_key",
        "circuit_short_name_x": "circuit_short_name",
        "country_code_x": "country_code",
        "country_key_x": "country_key",
        "country_name_x": "country_name",
        "date_start_x": "date_start",
        "gmt_offset_x": "gmt_offset",
        "location_x": "location",
        "year_x": "year",
        "date_x": "date"
    })
    df = df.drop_duplicates()

    df.compute().to_csv("merged_openf1_data_9.csv", index=False)

    return df

merge_openf1_data()


