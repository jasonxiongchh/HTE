import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score,mean_absolute_error, accuracy_score,confusion_matrix
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

bdebug_mode = False
# Step 0: Seperate the data
if not bdebug_mode:
    # read data
    df_weather = pd.read_csv(
        r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Climate_HourlyWeather.csv",
        parse_dates=['Date/Time']
    )

    # rename columns
    df_weather.rename(columns={'Date/Time': 'time', 'Temp (C)': 'Temp'}, inplace=True)

    # type convert
    df_weather["time"] = pd.to_datetime(df_weather["time"])

    # set time as index
    df_weather.set_index("time", inplace=True)

    # generate full range of time
    full_range = pd.date_range(start=df_weather.index.min(), end=df_weather.index.max(), freq="T")

    # resample to minute frequency
    minute_df = pd.DataFrame(index=full_range)
    minute_df.index.name = "time"
    minute_df = minute_df.join(df_weather, how="left")

    # forward fill temperature data
    minute_df["Temp"] = minute_df["Temp"].fillna(method="ffill")

    # forward fill weather data
    minute_df["Weather"] = minute_df["Weather"].fillna(method="ffill")

    # auto fill the missing data
    minute_df['Weather'] = minute_df['Weather'].fillna('Unknown')
    unique_weather_types = minute_df['Weather'].unique()
    weather_mapping = {weather: idx for idx, weather in enumerate(unique_weather_types)}
    minute_df['Weather_numeric'] = minute_df['Weather'].map(weather_mapping)

    # save the minute data
    minute_df = minute_df.reset_index()[['time', 'Weather', 'Weather_numeric', 'Temp']]
    minute_weather_output_path = "minute_weather_data.csv"
    minute_df.to_csv(minute_weather_output_path, index=False)
    print(f" {minute_weather_output_path}")

    # read data
    df_electricity = pd.read_csv(
        r"F:\uqstudy\project\code\electricity_usage_monitoring\Electricity_HTE_1.csv",
        parse_dates=['time']
    )

    # feature engineering
    df_electricity['time_minutes'] = (df_electricity['time'] - df_electricity['time'].min()).dt.total_seconds() // 60
    df_electricity['hour'] = df_electricity['time'].dt.hour
    df_electricity['weekday'] = df_electricity['time'].dt.dayofweek
    df_electricity['year'] = df_electricity['time'].dt.year
    df_electricity['month'] = df_electricity['time'].dt.month

    def get_season(month):
        if month in [12, 1, 2]:
            return 4  
        elif month in [3, 4, 5]:
            return 1  
        elif month in [6, 7, 8]:
            return 2  
        else:
            return 3  

    df_electricity['season'] = df_electricity['time'].dt.month.apply(get_season)

    df_weather_resampled = minute_df.set_index('time').resample('min').ffill().reset_index()

    # merge data
    df_merged = pd.merge(df_electricity, df_weather_resampled, on='time', how='left')

    # fill missing values
    df_merged['Temp'] = df_merged['Temp'].fillna(method='ffill')
    df_merged['label'] = 0

    # columns to keep
    columns_to_keep = ['watt', 'time', 'Temp', 'time_minutes', 'year', 'month','hour', 'weekday', 'season', 'Weather_numeric', 'Weather','label']
    df_merged = df_merged[[col for col in columns_to_keep if col in df_merged.columns]]

    # save data
    output_dir = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data"
    output_filename = "preprocessed_data_HTE.csv"
    output_path = os.path.join(output_dir, output_filename)
    df_merged.to_csv(output_path, index=False, na_rep='NA')
    print(f"Merged data saved to {output_path}")


    # data devide
    train_data_end = "2013-10-02 23:59:00"
    eval_data_end = "2014-1-01 23:59:00"
    pred_data_end = "2014-03-30 23:59:00"

    data = pd.read_csv(output_path, parse_dates=['time'])
    # **3. Data Devide**
    # date(1-31), month, year, weekdate,time, temp, weather(test later), season
    eval_data_end_ts = pd.Timestamp(eval_data_end)
    test_oneday = eval_data_end_ts + pd.Timedelta(days=1) 
    test_oneweek = eval_data_end_ts + pd.Timedelta(weeks=1)
    test_onemonth = eval_data_end_ts + pd.DateOffset(months=1)
    train_data = data[data['time'] <= train_data_end]
    eval_data = data[(data['time'] > train_data_end) & (data['time'] <= eval_data_end)]
    pred_data = data[(data['time'] > eval_data_end) & (data['time'] <= pred_data_end)]
    test_data = data[data['time'] >= eval_data_end]
    test_data_oneday = data[(data['time'] > eval_data_end) & (data['time'] <= test_oneday)]
    test_data_oneweek = data[(data['time'] > eval_data_end) & (data['time'] <= test_oneweek)]
    test_data_onemonth = data[(data['time'] > eval_data_end) & (data['time'] <= test_onemonth)]


    train_data.to_csv(os.path.join(output_dir, "Electricity_HTE_trainData.csv"), index=False)
    eval_data.to_csv(os.path.join(output_dir, "Electricity_HTE_evalData.csv"), index=False)
    pred_data.to_csv(os.path.join(output_dir, "Electricity_HTE_predData.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "Electricity_HTE_testData.csv"), index=False)
    test_data_oneday.to_csv(os.path.join(output_dir, "Electricity_HTE_testData_oneday.csv"), index=False)
    test_data_oneweek.to_csv(os.path.join(output_dir, "Electricity_HTE_testData_oneweek.csv"), index=False)
    test_data_onemonth.to_csv(os.path.join(output_dir, "Electricity_HTE_testData_onemonth.csv"), index=False)


    base_path = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data'
    file_name = 'Electricity_HTE_testData_oneday.csv' 
    modi_path = os.path.join(base_path, file_name)
    df_modi = pd.read_csv(modi_path)
    df_modi['time'] = pd.to_datetime(df_modi['time'])
    # create a copy of the DataFrame
    df_modification = df_modi.copy()
    df_modification['modified'] = 0

    # determine the start and end time
    start_time = df_modi['time'].min()
    end_time = df_modi['time'].max()

    count = 0
    num_modifications = 10 # placeholder for the number of modifications

  
    df_modification = df_modification.sort_values(by="time").reset_index(drop=True)

    # generate random time
    count = 0
    while count < num_modifications:
        # generate a random time
        random_minutes = np.random.randint(0, int((end_time - start_time).total_seconds() // 60))
        random_time = start_time + pd.to_timedelta(random_minutes, unit="m")

        # choose a random index to modify
        mask = (df_modification['time'] >= random_time) & (df_modification['time'] < (random_time + pd.Timedelta(minutes=1)))
        if mask.any():  
            random_index = df_modification[mask].sample(n=1).index[0]
            
            original_value = df_modification.loc[random_index, 'watt']  
            random_value = original_value * np.random.uniform(20, 50) 
            
            # edit data
            df_modification.loc[random_index, 'watt'] = random_value
            df_modification.loc[random_index, 'modified'] = 1  
            count += 1

    # save modified CSV
    output_path = os.path.join(output_dir, "Electricity_HTE_testData_oneday_modification.csv")
    df_modification.to_csv(output_path, index=False)
    df_modification = df_modification.drop(columns=['modified'])


    base_path_1 = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data'
    file_name_1 = 'Electricity_HTE_testData_oneweek.csv' 
    modi_path_1 = os.path.join(base_path_1, file_name_1)
    df_modi_1 = pd.read_csv(modi_path_1)
    df_modi_1['time'] = pd.to_datetime(df_modi_1['time'])
    # create a copy of the DataFrame
    df_modification_1 = df_modi_1.copy()
    df_modification_1['modified'] = 0

    # determine the start and end time
    start_time_1 = df_modi_1['time'].min()
    end_time_1 = df_modi_1['time'].max()

    count_1 = 0
    num_modifications_1 = 30 # placeholder for the number of modifications

  
    df_modification_1 = df_modification_1.sort_values(by="time").reset_index(drop=True)

    # generate random time
    count_1 = 0
    while count_1 < num_modifications_1:
        # generate a random time
        random_minutes_1 = np.random.randint(0, int((end_time_1 - start_time_1).total_seconds() // 60))
        random_time_1 = start_time_1 + pd.to_timedelta(random_minutes_1, unit="m")

        # choose a random index to modify
        mask_1 = (df_modification_1['time'] >= random_time_1) & (df_modification_1['time'] < (random_time_1 + pd.Timedelta(minutes=1)))
        if mask_1.any():  
            random_index_1 = df_modification_1[mask_1].sample(n=1).index[0]
            
            original_value_1 = df_modification_1.loc[random_index, 'watt']  
            random_value_1 = original_value_1 * np.random.uniform(20, 50) 
            
            # edit data
            df_modification_1.loc[random_index_1, 'watt'] = random_value_1
            df_modification_1.loc[random_index_1, 'modified'] = 1  
            count_1 += 1

    # save modified CSV
    output_path_1 = os.path.join(output_dir, "Electricity_HTE_testData_oneweek_modification.csv")
    df_modification_1.to_csv(output_path_1, index=False)
    df_modification_1 = df_modification_1.drop(columns=['modified'])


    base_path_2 = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data'
    file_name_2 = 'Electricity_HTE_testData_onemonth.csv' 
    modi_path_2 = os.path.join(base_path_2, file_name_2)
    df_modi_2 = pd.read_csv(modi_path_2)
    df_modi_2['time'] = pd.to_datetime(df_modi_2['time'])
    # create a copy of the DataFrame
    df_modification_2 = df_modi_2.copy()
    df_modification_2['modified'] = 0

    # determine the start and end time
    start_time_2 = df_modi_2['time'].min()
    end_time_2 = df_modi_2['time'].max()

    count_2 = 0
    num_modifications_2 = 50 # placeholder for the number of modifications

  
    df_modification_2 = df_modification_2.sort_values(by="time").reset_index(drop=True)

    # generate random time
    count_2 = 0
    while count_2 < num_modifications_2:
        # generate a random time
        random_minutes_2 = np.random.randint(0, int((end_time_2 - start_time_2).total_seconds() // 60))
        random_time_2 = start_time_2 + pd.to_timedelta(random_minutes_2, unit="m")

        # choose a random index to modify
        mask_2 = (df_modification_2['time'] >= random_time_2) & (df_modification_2['time'] < (random_time_2 + pd.Timedelta(minutes=1)))
        if mask_2.any():  
            random_index_2 = df_modification_2[mask_2].sample(n=1).index[0]
            
            original_value_2 = df_modification_2.loc[random_index, 'watt']  
            random_value_2 = original_value_2 * np.random.uniform(20, 50) 
            
            # edit data
            df_modification_2.loc[random_index_2, 'watt'] = random_value_2
            df_modification_2.loc[random_index_2, 'modified'] = 1  
            count_2 += 1

    # save modified CSV
    output_path_2 = os.path.join(output_dir, "Electricity_HTE_testData_onemonth_modification.csv")
    df_modification_2.to_csv(output_path_2, index=False)
    df_modification_2 = df_modification_2.drop(columns=['modified'])



        

    # min-max normalization
    columns_to_normalize = ['watt', 'Temp', 'time_minutes', 'year', 'month', 'hour', 'weekday', 'season', 'Weather_numeric']
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(train_data[columns_to_normalize])

    def normalize_and_save(data, columns, scaler, file_path):
        normalized_data = pd.DataFrame(scaler.transform(data[columns]), columns=columns)
        normalized_data = pd.concat([data.drop(columns + ['Weather', 'time'], axis=1).reset_index(drop=True), normalized_data], axis=1)
        normalized_data.to_csv(file_path, index=False)

    normalize_and_save(train_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_trainData_normalized.csv"))
    normalize_and_save(eval_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_evalData_normalized.csv"))
    normalize_and_save(pred_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_predData_normalized.csv"))
    normalize_and_save(test_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_normalized.csv"))
    normalize_and_save(test_data_oneday, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneday_normalized.csv"))
    normalize_and_save(df_modification, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneday_modification_normalized_20.csv"))
    normalize_and_save(df_modification_1, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneweek_modification_normalized_20.csv"))
    normalize_and_save(df_modification_2, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_onemonth_modification_normalized_20.csv"))
    normalize_and_save(test_data_oneweek, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneweek_normalized.csv"))
    normalize_and_save(test_data_onemonth, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_onemonth_normalized.csv"))

    print("Data normalization and saving complete.")


# path
file_path_train_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_trainData_normalized.csv"
file_path_eval_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_evalData_normalized.csv"
file_path_pred_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_predData_normalized.csv"
file_path_test_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_normalized.csv"
# file_path_test_normalized_oneday = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_oneday_normalized.csv"
file_path_test_normalized_oneday_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_oneday_modification_normalized_20.csv"
file_path_test_normalized_oneweek_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_oneweek_modification_normalized_20.csv"
file_path_test_normalized_onemonth_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_onemonth_modification_normalized_20.csv"
# file_path_test_normalized_oneweek = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_oneweek_normalized.csv"
# file_path_test_normalized_onemonth = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_onemonth_normalized.csv"
train_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\reshaped\Electricity_HTE_trainData_normalized_reshaped_20_MORE.csv"
val_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\reshaped\Electricity_HTE_evalData_normalized_reshaped_20_MORE.csv"
train_target_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\reshaped\Electricity_HTE_trainData_normalized_target_reshaped_20_MORE.csv"
val_target_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\reshaped\Electricity_HTE_evalData_normalized_target_reshaped_20_MORE.csv"




# read data
data_normalized_train = pd.read_csv(file_path_train_normalized)
data_normalized_eval = pd.read_csv(file_path_eval_normalized)
data_normalized_pred = pd.read_csv(file_path_pred_normalized)
data_normalized_test = pd.read_csv(file_path_test_normalized)
# data_normalized_test_oneday = pd.read_csv(file_path_test_normalized_oneday)
data_normalized_test_oneday_modification = pd.read_csv(file_path_test_normalized_oneday_modification)
data_normalized_test_oneweek_modification = pd.read_csv(file_path_test_normalized_oneweek_modification)
data_normalized_test_onemonth_modification = pd.read_csv(file_path_test_normalized_onemonth_modification)
# data_normalized_test_oneweek = pd.read_csv(file_path_test_normalized_oneweek)
# data_normalized_test_onemonth = pd.read_csv(file_path_test_normalized_onemonth)

# setting

time_steps = 20
features = ['watt', 'Temp', 'time_minutes', 'year', 'month', 'hour', 'weekday', 'season', 'Weather_numeric','label']

for data in [data_normalized_train, data_normalized_eval, 
             data_normalized_pred, data_normalized_test, 
             data_normalized_test_oneday_modification, 
             data_normalized_test_oneweek_modification, data_normalized_test_onemonth_modification]:
    data['power_change_rate'] = data['watt'].diff().fillna(0)
    data['watt_hour_interaction'] = data['watt'] * data['hour']
    data['watt_season_interaction'] = data['watt'] * data['season']
    data['watt_weather_interaction'] = data['watt'] * data['Weather_numeric']
    data['watt_month_interaction'] = data['watt'] * data['month']
    data['watt_weekday_interaction'] = data['watt'] * data['weekday']
    data['watt_Temp_interaction']=data['watt']*data['Temp']
    data['watt_hour_weekday_interaction'] = data['watt'] * data['hour'] * data['weekday']
    data['watt_hour_month_interaction'] = data['watt'] * data['hour'] * data['month']
    data['watt_hour_season_interaction'] = data['watt'] * data['hour'] * data['season']
    data['watt_hour_weather_interaction'] = data['watt'] * data['hour'] * data['Weather_numeric']
    data['watt_hour_Temp_interaction']=data['watt']*data['hour']*data['Temp']
    data['watt_hour_weekday_month_interaction'] = data['watt'] * data['hour'] * data['weekday'] * data['month']
    data['watt_hour_weekday_season_interaction'] = data['watt'] * data['hour'] * data['weekday'] * data['season']
    data['watt_moving_avg'] = data['watt'].rolling(window=5).mean().fillna(data['watt'])

# define different features
interaction_features = ['power_change_rate','watt_hour_interaction',
                        'watt_season_interaction','watt_weather_interaction',
                        'watt_month_interaction', 'watt_weekday_interaction',
                        'watt_Temp_interaction','watt_hour_weekday_interaction',
                        'watt_hour_month_interaction','watt_hour_season_interaction',
                        'watt_hour_weather_interaction','watt_hour_Temp_interaction',
                        'watt_hour_weekday_month_interaction','watt_hour_weekday_season_interaction',
                        'watt_moving_avg']
final_features = features + interaction_features

# data_normalized_eval = data_normalized_eval.drop(columns=['label'])
# data_normalized_pred = data_normalized_pred.drop(columns=['label'])
# data_normalized_test = data_normalized_test.drop(columns=['label'])
# # data_normalized_test_oneday = data_normalized_test_oneday.drop(columns=['label'])
# data_normalized_test_oneday_modification = data_normalized_test_oneday_modification.drop(columns=['label'])
# data_normalized_test_oneweek_modification = data_normalized_test_oneweek_modification.drop(columns=['label'])
# data_normalized_test_onemonth_modification = data_normalized_test_onemonth_modification.drop(columns=['label'])



# reshape data
def reshape_data(data, time_steps, features_count):
    """
    Reshape time series data into LSTM-compatible format.

    Parameters:
    - data: np.array, shape (num_samples, features_count)
        The original time series data.
    - time_steps: int
        The number of past time steps to use as input for each sample.
    - features_count: int
        The number of features per time step.

    Returns:
    - reshaped_data: np.array, shape (num_samples - time_steps + 1, time_steps, features_count)
        The reshaped dataset, suitable for LSTM input.

    Raises:
    - ValueError: If the dataset has fewer samples than the required `time_steps`.
    """

    # Ensure there are enough data samples to form at least one window
    if len(data) < time_steps:
        raise ValueError(f"Data samples ({len(data)}) are less than the required time steps ({time_steps}).")

    # Preallocate memory for reshaped data
    reshaped_data = np.zeros((len(data) - time_steps + 1, time_steps, features_count), dtype=np.float64)

    # Create sliding windows
    for i in range(len(data) - time_steps + 1):
        reshaped_data[i] = data[i:i + time_steps]

    return reshaped_data


# save and load
def save_reshaped_data(reshaped_data, pfeatures, time_steps, save_path):
    samples = reshaped_data.shape[0]
    flattened_data = reshaped_data.reshape(samples, -1)
    column_names = [f"{feature}_t{t}" for t in range(time_steps) for feature in pfeatures]
    reshaped_df = pd.DataFrame(flattened_data, columns=column_names)
    reshaped_df.to_csv(save_path, index=False)

def load_reshaped_data(load_path, pfeatures, time_steps):
    reshaped_df = pd.read_csv(load_path)
    samples = reshaped_df.shape[0]
    num_features = len(pfeatures)
    original_shape = (samples, time_steps, num_features)
    reshaped_data = reshaped_df.values.reshape(original_shape)
    return reshaped_data

isDataChanged = False

if isDataChanged:
    train_data_for_model = reshape_data(data_normalized_train[final_features].values, time_steps, len(final_features))
    val_data_for_model = reshape_data(data_normalized_eval[final_features].values, time_steps, len(final_features))
    # y_train = data_normalized_train[['watt']].values
    # y_val = data_normalized_eval[['watt']].values
    #  Explanation:
    # - We extract the `.values` from `data_normalized_train[final_features]` to get a NumPy array.
    # - This ensures the data is in a format suitable for `reshape_data()`, which expects a NumPy array.
    # - `reshape_data()` organizes the data into `(num_samples, time_steps, features_count)`,
    #   where `num_samples = len(data) - time_steps + 1`.
    # - We are preparing the dataset for LSTM training, **not making a batch prediction** at this step.
    #
    #  Why not use batch_size here?
    # - At this stage, we are **preparing the full training dataset**, not running inference.
    # - LSTM expects an input shape of `(batch_size, time_steps, features_count)`, but batch size is
    #   only needed **at prediction time** (when making a single forecast).
    # - Here, we are structuring the dataset, not performing a single prediction, so `batch_size` is
    #   not relevant.
    save_reshaped_data(train_data_for_model, final_features, time_steps, train_save_path)
    save_reshaped_data(val_data_for_model, final_features, time_steps, val_save_path)
    # save_reshaped_data(y_train, ['watt'], 1, train_target_save_path)
    # save_reshaped_data(y_val, ['watt'], 1, val_target_save_path)


else:
    train_data_for_model = load_reshaped_data(train_save_path, final_features, time_steps)
    val_data_for_model = load_reshaped_data(val_save_path, final_features, time_steps)
    # y_train = load_reshaped_data(train_target_save_path, ['watt'], 1)
    # y_val = load_reshaped_data(val_target_save_path, ['watt'], 1)

# target
y_train = data_normalized_train['watt'].values[time_steps-1:len(train_data_for_model) + time_steps-1].reshape(-1, 1)
y_val = data_normalized_eval['watt'].values[time_steps-1:len(val_data_for_model) + time_steps-1].reshape(-1, 1)

# define model
def build_lstm_model(time_steps, features_count):
    input_shape = (time_steps, features_count)
    input_layer = Input(shape=input_shape)

    # CNN layer
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = Dropout(0.2)(x)

    # LSTM layer
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # Fully Connected Layer
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


bload_model = True
# training
if not bload_model:
    cnn_lstm = build_lstm_model(time_steps,len(final_features))
    cnn_lstm.summary()

    history = cnn_lstm.fit(
        train_data_for_model, y_train,
        batch_size=64,
        epochs=10,
        # validation_data=(val_data_for_model, y_val)
    )

    cnn_lstm.save('cnn_lstm_IF_TS20_with_label.keras')
    # loss = cnn_lstm.evaluate(val_data_for_model, y_val)
    # print(f"Evaluation Loss: {loss[0]:.4f}, Evaluation MAE: {loss[1]:.4f}")
else:
    cnn_lstm = load_model('cnn_lstm_IF_TS20_with_label.keras', compile=False)
    cnn_lstm.compile(optimizer='adam', loss='mse')


test_importance= False

if test_importance:
    # Step 4: Feature Importance using Permutation Importance

    # MAE（baseline）
    y_pred = cnn_lstm.predict(val_data_for_model)
    baseline_mae = mean_absolute_error(y_val, y_pred)

    # use permutation
    feature_importance = []

    for i in range(val_data_for_model.shape[2]):
        X_permuted = val_data_for_model.copy()
        np.random.shuffle(X_permuted[:, :, i])  
        
        y_pred_permuted = cnn_lstm.predict(X_permuted)
        permuted_mae = mean_absolute_error(y_val, y_pred_permuted)

        importance_score = permuted_mae - baseline_mae
        feature_importance.append(importance_score)

    importance_df = pd.DataFrame({
        'Feature': final_features,
        'Importance_Score': feature_importance
    }).sort_values(by='Importance_Score', ascending=False)

    output_dir_a = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data"
    output_filename_a = 'feature_importance_permutation.csv'
    output_path_a = os.path.join(output_dir_a, output_filename_a)
    importance_df.to_csv(output_path_a, index=False)
    print(f"Feature importance data saved to {output_path_a}")



compare_result = True

if compare_result:

    result  = cnn_lstm.predict(val_data_for_model)
    result = result.astype(np.float64)
    result_actul = data_normalized_eval['watt'].values
    result_actul = result_actul[19:]
    mae = mean_absolute_error(result_actul, result)

    plt.figure(figsize=(10, 4))
    plt.plot(result_actul, label='True')
    plt.plot(result, label='Predicted')
    plt.title('Model Prediction vs True Values')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========================================================================================================================================================================
# normalized_eval_subset = data_normalized_eval[columns_to_normalize].iloc[time_steps - 1:time_steps - 1 + len(result)].copy()
# predicted_normalized = normalized_eval_subset.copy()
# predicted_normalized['watt'] = result.flatten()

# real_values = scaler.inverse_transform(normalized_eval_subset)
# predicted_values = scaler.inverse_transform(predicted_normalized)

# df_real = pd.DataFrame(real_values, columns=columns_to_normalize)
# df_pred = pd.DataFrame(predicted_values, columns=columns_to_normalize)

# # ==========DataFrame==========
# comparison_df = pd.DataFrame({
#     'time_minutes': df_real['time_minutes'],
#     'year': df_real['year'].astype(int),
#     'month': df_real['month'].astype(int),
#     'hour': df_real['hour'].astype(int),
#     'weekday': df_real['weekday'].astype(int),
#     'watt': df_real['watt'],
#     'watt_predicted': df_pred['watt']
# })

# # # culculate anomaly label
# # comparison_df['anomaly_label'] = (np.abs(comparison_df['watt'] - comparison_df['watt_predicted']) > 0.2).astype(int)

# comparison_df['relative_error'] = np.abs(comparison_df['watt'] - comparison_df['watt_predicted']) / (np.abs(comparison_df['watt']) + 1e-6)

# # set threshold
# threshold = 0.20 
# comparison_df['anomaly_label'] = (comparison_df['relative_error'] > threshold).astype(int)
# # ====================
# output_compare_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\cnn_lstm_prediction_comparison_with_label.csv"
# comparison_df.to_csv(output_compare_path, index=False)
# print(f"saved as csv: {output_compare_path}")


# plt.figure(figsize=(12, 5))
# plt.plot(comparison_df['watt'].values, label='True', linewidth=1)
# plt.plot(comparison_df['watt_predicted'].values, label='Predicted', linewidth=1)

# anomaly_indices = comparison_df[comparison_df['anomaly_label'] == 1].index
# plt.scatter(anomaly_indices, comparison_df.loc[anomaly_indices, 'watt_predicted'], 
#             color='red', label='Anomaly', marker='o', s=30, zorder=5)

# plt.title('Model Prediction vs True Values with Anomalies')
# plt.xlabel('Time step')
# plt.ylabel('Watt')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()





# ========================================================================================================================================================================
# true_labels = data_normalized_eval['label'].values[time_steps - 1:time_steps - 1 + len(comparison_df)]
# pred_labels = comparison_df['anomaly_label'].values

# precision = precision_score(true_labels, pred_labels)
# recall = recall_score(true_labels, pred_labels)
# f1 = f1_score(true_labels, pred_labels)
# accuracy = accuracy_score(true_labels, pred_labels)

# print(f" Precision: {precision:.4f}")
# print(f" Recall:    {recall:.4f}")
# print(f" F1 Score:  {f1:.4f}")
# print(f" Accuracy:  {accuracy:.4f}")

# # 生成混淆矩阵
# cm = confusion_matrix(true_labels, pred_labels)
# labels = ['Normal', 'Anomaly']

# # 绘图
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.tight_layout()
# plt.show()

# Step 5: Detect anomalies using the trained model

errors_mse = []
errors_mae =[]
timestamps = []
anomaly_scores_list_mse = []
anomaly_labels_list_mse = []
anomaly_scores_list_mae = []
anomaly_labels_list_mae = []



# define parameters for Isolation Forest
iso_forest_params_mse = {
    'n_estimators': 100,
    'contamination': 0.001,
    'random_state': 42
}

iso_forest_params_mae = {
    'n_estimators': 100,
    'contamination': 0.001,
    'random_state': 42
}


IF_is_well_trained = True

if not IF_is_well_trained:
    start_time = datetime.now()
    print(f"Start calculating training errors at: {start_time}")
    
    train_predictions = cnn_lstm.predict(train_data_for_model)
    train_predictions = train_predictions.astype(np.float64)
    train_predictions_pre = train_predictions.flatten()
    train_predictions_pre = train_predictions_pre[:len(data_normalized_train) - time_steps + 1]
    
    train_errors_mse = []
    train_errors_mae = []
    for i in range(len(train_predictions_pre) - time_steps + 1):
        # window_real = y_train[i : i + time_steps].flatten()
        window_real = data_normalized_train.iloc[i : i + time_steps]['watt'].values 
        window_pred = train_predictions_pre[i : i + time_steps]
        
        window_error_mse = np.mean((window_real - window_pred) ** 2)  # MSE
        window_mae = np.mean(np.abs(window_real - window_pred))
        train_errors_mse.append(window_error_mse)
        train_errors_mae.append(window_mae)
    
    train_errors_mse = np.array(train_errors_mse)
    train_errors_mae = np.array(train_errors_mae)

    iso_forest_mse = IsolationForest(**iso_forest_params_mse)
    iso_forest_mae = IsolationForest(**iso_forest_params_mae)
    iso_forest_mse.fit(train_errors_mse.reshape(-1, 1))
    iso_forest_mae.fit(train_errors_mae.reshape(-1, 1))

    print(f"Isolation Forest model trained at: {datetime.now()}")

    with open('iso_forest_model_S1_TS20_mse.pkl', 'wb') as f:
        pickle.dump(iso_forest_mse, f)
    with open('iso_forest_model_S1_TS20_mae.pkl', 'wb') as f:
        pickle.dump(iso_forest_mae, f)
    

    print(f"Isolation Forest model saved at: {datetime.now()}")
else:
    with open('iso_forest_model_S1_TS20_mse.pkl', 'rb') as f:
        iso_forest_mse = pickle.load(f)
    with open('iso_forest_model_S1_TS20_mae.pkl', 'rb') as f:
        iso_forest_mae = pickle.load(f)

    print(f"Isolation Forest model loaded at: {datetime.now()}")



# plt.figure(figsize=(10, 6))
# plt.plot(train_errors_mse, marker='o', linestyle='-', color='b', label="MSE Errors")
# plt.xlabel('Sample Index')
# plt.ylabel('MSE Error')
# plt.title('Training Errors Visualization')
# plt.legend()
# plt.grid(True)
# plt.show()

# data_normalized_train = data_normalized_train[:int(len(data_normalized_train) * 0.01)]


# for i in range(len(data_normalized_train) - time_steps + 1):  # make predictions
#     loop_start_time = datetime.now()

#     window = data_normalized_train.iloc[i : i + time_steps][final_features].values

#     prediction = cnn_lstm.predict(window.reshape(1, time_steps, len(final_features)), verbose=0)
#     prediction = prediction.astype(np.float64)

#     prediction_time = datetime.now()
#     print(f"Prediction completed at: {prediction_time}, Time taken: {prediction_time - loop_start_time}")

#     # calculate reconstruction error
#     window_actual = data_normalized_train.iloc[i : i + time_steps]['watt'].values
#     window_pred = prediction.flatten()  # what the model predicted

#     # calculate reconstruction error
#     reconstruction_error = np.mean((window_actual - window_pred) ** 2)

#     errors.append(reconstruction_error)
#     timestamps.append(data_normalized_train.iloc[i]['time_minutes'])

#     # calculate anomaly score
#     anomaly_score = iso_forest.decision_function([[reconstruction_error]])[0]
#     anomaly_label = iso_forest.predict([[reconstruction_error]])[0] == -1

#     detection_time = datetime.now()
#     print(f"Anomaly detection completed at: {detection_time}, Time taken: {detection_time - prediction_time}")

#     anomaly_scores_list.append(anomaly_score)
#     anomaly_labels_list.append(anomaly_label)

#     print(f"Current time: {detection_time}, Anomaly detected: {anomaly_label}, Loop duration: {detection_time - loop_start_time}")





for i in range(len(data_normalized_test_oneday_modification) - time_steps + 1):  # make predictions
    loop_start_time = datetime.now()

    window = data_normalized_test_oneday_modification.iloc[i : i + time_steps][final_features].values

    prediction = cnn_lstm.predict(window.reshape(1, time_steps, len(final_features)), verbose=0)
    prediction = prediction.astype(np.float64)

    prediction_time = datetime.now()
    print(f"Prediction completed at: {prediction_time}, Time taken: {prediction_time - loop_start_time}")

    # calculate reconstruction error
    window_actual = data_normalized_test_oneday_modification.iloc[i : i + time_steps]['watt'].values
    window_pred = prediction.flatten()  # what the model predicted

    # calculate reconstruction error
    reconstruction_error = np.mean((window_actual - window_pred) ** 2)
    Absolute_mae = np.mean(np.abs(window_actual - window_pred))

    errors_mse.append(reconstruction_error)
    errors_mae.append(Absolute_mae)
    timestamps.append(data_normalized_test_oneday_modification.iloc[i]['time_minutes'])

    # calculate anomaly score
    anomaly_score_mse = iso_forest_mse.decision_function([[reconstruction_error]])[0]
    anomaly_label_mse = iso_forest_mse.predict([[reconstruction_error]])[0] == -1
    anomaly_score_mae = iso_forest_mae.decision_function([[Absolute_mae]])[0]
    anomaly_label_mae = iso_forest_mae.predict([[Absolute_mae]])[0] == -1

    detection_time = datetime.now()
    print(f"Anomaly detection completed at: {detection_time}, Time taken: {detection_time - prediction_time}")

    anomaly_scores_list_mse.append(anomaly_score_mse)
    anomaly_labels_list_mse.append(anomaly_label_mse)
    anomaly_scores_list_mae.append(anomaly_score_mae)
    anomaly_labels_list_mae.append(anomaly_label_mae)

    print(f"Current time: {detection_time}, Loop duration: {detection_time - loop_start_time}")




# create dataframe
error_df_mse = pd.DataFrame({
    'Timestamp': timestamps,
    'Reconstruction Error': errors_mse,
    'Anomaly Score_mse': anomaly_scores_list_mse,
    'Is Anomaly_mse': anomaly_labels_list_mse
})

error_df_mae = pd.DataFrame({
    'Timestamp': timestamps,
    'Absolute MAE': errors_mae,
    'Anomaly Score_mae': anomaly_scores_list_mae,
    'Is Anomaly_mae': anomaly_labels_list_mae
})


# path
csv_file_path_mse = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\anomaly_detection_results_20_mse.csv"
csv_file_path_mae = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\anomaly_detection_results_20_mae.csv" 

# save to csv
error_df_mse.to_csv(csv_file_path_mse, index=False)
error_df_mae.to_csv(csv_file_path_mae, index=False)

anomalies_mse = error_df_mse[error_df_mse['Is Anomaly_mse']]
anomalies_mae = error_df_mae[error_df_mae['Is Anomaly_mae']]

# draw the figure
plt.figure(figsize=(12, 12))

# **Subplot 1: Reconstruction Error**
plt.subplot(3, 1, 1)
plt.plot(error_df_mse['Timestamp'], error_df_mse['Reconstruction Error'], label='Reconstruction Error (MSE)', color='blue')
plt.plot(error_df_mae['Timestamp'], error_df_mae['Absolute MAE'], label='Absolute Error (MAE)', color='green')  

# draw MSE anomalies
plt.scatter(anomalies_mse['Timestamp'], anomalies_mse['Reconstruction Error'], color='red', label='MSE Anomalies', zorder=3, marker='o')

# draw MAE anomalies
plt.scatter(anomalies_mae['Timestamp'], anomalies_mae['Absolute MAE'], color='yellow', label='MAE Anomalies', zorder=3, marker='x')

plt.title('Reconstruction Error Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Reconstruction Error')
plt.xticks(rotation=45)
plt.legend()

# **Subplot 2: Anomaly Scores**
plt.subplot(3, 1, 2)
plt.plot(error_df_mse['Timestamp'], error_df_mse['Anomaly Score_mse'], label='Anomaly Score (MSE)', color='green')
plt.plot(error_df_mae['Timestamp'], error_df_mae['Anomaly Score_mae'], label='Anomaly Score (MAE)', color='purple')
plt.scatter(anomalies_mse['Timestamp'], anomalies_mse['Anomaly Score_mse'], color='red', label='MSE Anomalies', zorder=3, marker='o')
plt.scatter(anomalies_mae['Timestamp'], anomalies_mae['Anomaly Score_mae'], color='orange', label='MAE Anomalies', zorder=3, marker='x')
plt.title('Anomaly Scores Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.xticks(rotation=45)
plt.legend()

# # **Subplot 3: Actual Power Consumption**
# plt.subplot(3, 1, 3)
# plt.plot(data_normalized_test_oneday_modification['time_minutes'], data_normalized_test_oneday_modification['watt'], label='Actual Power Consumption', color='purple')

# # find the corresponding watt values for anomalies
# mse_anomalies_watt = data_normalized_test_oneday_modification.loc[
#     data_normalized_test_oneday_modification['time_minutes'].isin(anomalies_mse['Timestamp']), 'watt'
# ]
# mae_anomalies_watt = data_normalized_test_oneday_modification.loc[
#     data_normalized_test_oneday_modification['time_minutes'].isin(anomalies_mae['Timestamp']), 'watt'
# ]

# plt.scatter(anomalies_mse['Timestamp'], mse_anomalies_watt, color='red', label='MSE Anomalies', zorder=3, marker='o')
# plt.scatter(anomalies_mae['Timestamp'], mae_anomalies_watt, color='yellow', label='MAE Anomalies', zorder=3, marker='x')

# plt.title('Power Consumption Over Time')
# plt.xlabel('Timestamp')
# plt.ylabel('Watt')
# plt.xticks(rotation=45)
# plt.legend()

# plt.tight_layout()
# plt.show(block=True)

shift_step = time_steps - 1

# find the index of anomalies in the shifted data
mse_anomalies_idx = data_normalized_test_oneday_modification.index[
    data_normalized_test_oneday_modification['time_minutes'].isin(anomalies_mse['Timestamp'])
]
mae_anomalies_idx = data_normalized_test_oneday_modification.index[
    data_normalized_test_oneday_modification['time_minutes'].isin(anomalies_mae['Timestamp'])
]

# make sure the index is valid
mse_anomalies_shifted_idx = [i + shift_step for i in mse_anomalies_idx if i + shift_step < len(data_normalized_test_oneday_modification)]
mae_anomalies_shifted_idx = [i + shift_step for i in mae_anomalies_idx if i + shift_step < len(data_normalized_test_oneday_modification)]

#  get time and watt
mse_anomalies_shifted_time = data_normalized_test_oneday_modification.iloc[mse_anomalies_shifted_idx]['time_minutes']
mse_anomalies_shifted_watt = data_normalized_test_oneday_modification.iloc[mse_anomalies_shifted_idx]['watt']

mae_anomalies_shifted_time = data_normalized_test_oneday_modification.iloc[mae_anomalies_shifted_idx]['time_minutes']
mae_anomalies_shifted_watt = data_normalized_test_oneday_modification.iloc[mae_anomalies_shifted_idx]['watt']

# draw shifted anomalies
plt.subplot(3, 1, 3)
plt.plot(data_normalized_test_oneday_modification['time_minutes'], data_normalized_test_oneday_modification['watt'], label='Actual Power Consumption', color='purple')

# # draw shifted anomalies
# plt.scatter(mse_anomalies_shifted_time, mse_anomalies_shifted_watt, color='red', label='MSE Anomalies (shifted)', zorder=3, marker='o')
# plt.scatter(mae_anomalies_shifted_time, mae_anomalies_shifted_watt, color='yellow', label='MAE Anomalies (shifted)', zorder=3, marker='x')

# create df
mse_df = pd.DataFrame({'time': mse_anomalies_shifted_time, 'watt': mse_anomalies_shifted_watt})
mae_df = pd.DataFrame({'time': mae_anomalies_shifted_time, 'watt': mae_anomalies_shifted_watt})

# concat
common_df = mse_df.merge(mae_df, on='time', suffixes=('_mse', '_mae'))

# print(common_df)
plt.scatter(common_df['time'], common_df['watt_mse'], color='red', label='Common Anomalies (MSE & MAE)', zorder=4, marker='D')

plt.title('Power Consumption Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Watt')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show(block=True)



