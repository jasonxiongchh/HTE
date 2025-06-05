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
    output_dir = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final"
    output_filename = "preprocessed_data_HTE.csv"
    output_path = os.path.join(output_dir, output_filename)
    df_merged.to_csv(output_path, index=False, na_rep='NA')
    print(f"Merged data saved to {output_path}")


    # data devide
    train_data_end = "2013-10-02 23:59:00"
    test_data_end = "2014-03-30 23:59:00"
    
    data = pd.read_csv(output_path, parse_dates=['time'])
    # **3. Data Devide**
    # date(1-31), month, year, weekdate,time, temp, weather(test later), season
    train_data_end_ts = pd.Timestamp(train_data_end)
    test_oneday = train_data_end_ts + pd.Timedelta(days=1) 
    test_oneweek = train_data_end_ts + pd.Timedelta(weeks=1)
    test_onemonth = train_data_end_ts + pd.DateOffset(months=1)
    train_data = data[data['time'] <= train_data_end]
    test_data = data[(data['time'] >= train_data_end) & (data['time'] <= test_data_end)]
    test_data_oneday = data[(data['time'] > train_data_end) & (data['time'] <= test_oneday)]
    test_data_onemonth = data[(data['time'] > train_data_end) & (data['time'] <= test_onemonth)]


    train_data.to_csv(os.path.join(output_dir, "Electricity_HTE_trainData.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "Electricity_HTE_testData.csv"), index=False)
    test_data_oneday.to_csv(os.path.join(output_dir, "Electricity_HTE_testData_oneday.csv"), index=False)
    test_data_onemonth.to_csv(os.path.join(output_dir, "Electricity_HTE_testData_onemonth.csv"), index=False)


    base_path = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data\\final'
    file_name = 'Electricity_HTE_testData_oneday.csv' 
    modi_path = os.path.join(base_path, file_name)
    df_modi = pd.read_csv(modi_path)
    df_modi['time'] = pd.to_datetime(df_modi['time'])

    df_modification = df_modi.copy()
    df_modification = df_modification.sort_values(by="time").reset_index(drop=True)

    df_modification['date'] = df_modification['time'].dt.date
    unique_dates = df_modification['date'].unique()

    for day in unique_dates:
        daily_data = df_modification[df_modification['date'] == day]
        if daily_data.empty:
            continue

        count = 0
        while count < 100:
            random_time = np.random.choice(daily_data['time'].values)
            mask = (df_modification['time'] >= random_time) & (df_modification['time'] < (pd.to_datetime(random_time) + pd.Timedelta(minutes=1)))

            if mask.any():
                random_index = df_modification[mask].sample(n=1, random_state=42).index[0]
                original_value = df_modification.loc[random_index, 'watt']
                random_value = original_value * np.random.uniform(10, 20)

                df_modification.loc[random_index, 'watt'] = random_value
                df_modification.loc[random_index, 'label'] = 1
                count += 1


    df_modification = df_modification.drop(columns=['date'])


    output_dir = base_path 
    output_path = os.path.join(output_dir, "Electricity_HTE_testData_oneday_modification.csv")
    df_modification.to_csv(output_path, index=False)


    base_path_2 = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data\\final'
    file_name_2 = 'Electricity_HTE_testData_onemonth.csv' 
    modi_path_2 = os.path.join(base_path_2, file_name_2)
    df_modi_2 = pd.read_csv(modi_path_2)
    df_modi_2['time'] = pd.to_datetime(df_modi_2['time'])

    df_modification_2 = df_modi_2.copy()
    df_modification_2 = df_modification_2.sort_values(by="time").reset_index(drop=True)

    df_modification_2['date'] = df_modification_2['time'].dt.date
    unique_dates_2 = df_modification_2['date'].unique()

    for day_2 in unique_dates_2:
        daily_data_2 = df_modification_2[df_modification_2['date'] == day_2]
        if daily_data_2.empty:
            continue

        count_2 = 0
        while count_2 < 100:
            random_time_2 = np.random.choice(daily_data_2['time'].values)
            mask_2 = (df_modification_2['time'] >= random_time_2) & (df_modification_2['time'] < (pd.to_datetime(random_time_2) + pd.Timedelta(minutes=1)))

            if mask_2.any():
                random_index_2 = df_modification_2[mask_2].sample(n=1, random_state=42).index[0]
                original_value_2 = df_modification_2.loc[random_index_2, 'watt']
                random_value_2 = original_value_2 * np.random.uniform(10, 20)

                df_modification_2.loc[random_index_2, 'watt'] = random_value_2
                df_modification_2.loc[random_index_2, 'label'] = 1
                count_2 += 1


    df_modification_2 = df_modification_2.drop(columns=['date'])

    output_dir_2 = base_path_2 
    output_path_2 = os.path.join(output_dir_2, "Electricity_HTE_testData_onemonth_modification.csv")
    df_modification_2.to_csv(output_path_2, index=False)


        

    # min-max normalization
    columns_to_normalize = ['watt', 'Temp', 'time_minutes', 'year', 'month', 'hour', 'weekday', 'season', 'Weather_numeric']
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(train_data[columns_to_normalize])

    def normalize_and_save(data, columns, scaler, file_path):
        normalized_data = pd.DataFrame(scaler.transform(data[columns]), columns=columns)
        normalized_data = pd.concat([data.drop(columns + ['Weather', 'time'], axis=1).reset_index(drop=True), normalized_data], axis=1)
        normalized_data.to_csv(file_path, index=False)

    normalize_and_save(train_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_trainData_normalized.csv"))
    normalize_and_save(test_data, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_normalized.csv"))
    normalize_and_save(test_data_oneday, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneday_normalized.csv"))
    normalize_and_save(df_modification, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_oneday_modification_normalized_20.csv"))
    normalize_and_save(df_modification_2, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_onemonth_modification_normalized_20.csv"))
    normalize_and_save(test_data_onemonth, columns_to_normalize, scaler, os.path.join(output_dir, "Electricity_HTE_testData_onemonth_normalized.csv"))

    print("Data normalization and saving complete.")

# path
file_path_train_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_trainData_normalized.csv"
file_path_test_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_testData_normalized.csv"
file_path_test_normalized_oneday_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_testData_oneday_modification_normalized_20.csv"
file_path_test_normalized_onemonth_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_testData_onemonth_modification_normalized_20.csv"
train_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_trainData_normalized_reshaped_20_MORE.csv"
test_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_testData_normalized_reshaped_20_MORE.csv"
test_day_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_test_day_Data_normalized_reshaped_20_MORE.csv"
test_month_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_test_month_Data_normalized_reshaped_20_MORE.csv"


# read data
data_normalized_train = pd.read_csv(file_path_train_normalized)
data_normalized_test = pd.read_csv(file_path_test_normalized)
# data_normalized_test_oneday = pd.read_csv(file_path_test_normalized_oneday)
data_normalized_test_oneday_modification = pd.read_csv(file_path_test_normalized_oneday_modification)
data_normalized_test_onemonth_modification = pd.read_csv(file_path_test_normalized_onemonth_modification)
# data_normalized_test_oneweek = pd.read_csv(file_path_test_normalized_oneweek)
# data_normalized_test_onemonth = pd.read_csv(file_path_test_normalized_onemonth)


time_steps = 20
features = ['watt', 'Temp', 'time_minutes', 'year', 'month', 'hour', 'weekday', 'season', 'Weather_numeric','label']


# reshape data
def reshape_data(data, time_steps, features_count):

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
    train_data_for_model = reshape_data(data_normalized_train[features].values, time_steps, len(features))
    test_data_for_model = reshape_data(data_normalized_test[features].values, time_steps, len(features))
    test_data_oneday_modification_for_model = reshape_data(data_normalized_test_oneday_modification[features].values, time_steps, len(features))
    test_data_onemonth_modification_for_model = reshape_data(data_normalized_test_onemonth_modification[features].values, time_steps, len(features))

    save_reshaped_data(train_data_for_model, features, time_steps, train_save_path)
    save_reshaped_data(test_data_for_model, features, time_steps, test_save_path)
    save_reshaped_data(test_data_oneday_modification_for_model, features, time_steps, test_day_save_path)
    save_reshaped_data(test_data_onemonth_modification_for_model, features, time_steps, test_day_save_path)


else:
    train_data_for_model = load_reshaped_data(train_save_path, features, time_steps)
    test_data_for_model = load_reshaped_data(test_save_path, features, time_steps)
    test_data_oneday_modification_for_model = load_reshaped_data(test_day_save_path, features, time_steps)
    test_data_onemonth_modification_for_model = load_reshaped_data(test_day_save_path, features, time_steps)
    
# target
y_train = data_normalized_train['watt'].values[time_steps-1:len(train_data_for_model) + time_steps-1].reshape(-1, 1)
y_val = data_normalized_test['watt'].values[time_steps-1:len(test_data_for_model) + time_steps-1].reshape(-1, 1)

# define model
def build_lstm_model_only(time_steps, features_count):
    input_shape = (time_steps, features_count)
    input_layer = Input(shape=input_shape)

    x = LSTM(20, return_sequences=False)(input_layer)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

bload_model = False
# training
if not bload_model:
    cnn_lstm = build_lstm_model_only(time_steps,len(features))
    cnn_lstm.summary()

    history = cnn_lstm.fit(
        train_data_for_model, y_train,
        batch_size=64,
        epochs=10,
    )

    cnn_lstm.save('lstm_40.keras')
else:
    cnn_lstm = load_model('lstm_40.keras', compile=False)
    cnn_lstm.compile(optimizer='adam', loss='mse')

test_importance= False

if test_importance:
    # Step 4: Feature Importance using Permutation Importance

    # MAE（baseline）
    y_pred = cnn_lstm.predict(test_data_for_model)
    baseline_mae = mean_absolute_error(y_val, y_pred)

    # use permutation
    feature_importance = []

    for i in range(test_data_for_model.shape[2]):
        X_permuted = test_data_for_model.copy()
        np.random.shuffle(X_permuted[:, :, i])  
        
        y_pred_permuted = cnn_lstm.predict(X_permuted)
        permuted_mae = mean_absolute_error(y_val, y_pred_permuted)

        importance_score = permuted_mae - baseline_mae
        feature_importance.append(importance_score)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance_Score': feature_importance
    }).sort_values(by='Importance_Score', ascending=False)

    output_dir_a = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final"
    output_filename_a = 'feature_importance_LSTM_permutation.csv'
    output_path_a = os.path.join(output_dir_a, output_filename_a)
    importance_df.to_csv(output_path_a, index=False)
    print(f"Feature importance data saved to {output_path_a}")

compare_result = True

if compare_result:

    result  = cnn_lstm.predict(test_data_onemonth_modification_for_model)
    result = result.astype(np.float64)
    result_actul = data_normalized_test_onemonth_modification['watt'].values
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

# ====================
normalized_eval_subset = data_normalized_test_onemonth_modification[columns_to_normalize].iloc[time_steps - 1:time_steps - 1 + len(result)].copy()
predicted_normalized = normalized_eval_subset.copy()
predicted_normalized['watt'] = result.flatten()

real_values = scaler.inverse_transform(normalized_eval_subset)
predicted_values = scaler.inverse_transform(predicted_normalized)

df_real = pd.DataFrame(real_values, columns=columns_to_normalize)
df_pred = pd.DataFrame(predicted_values, columns=columns_to_normalize)

# ==========DataFrame==========
comparison_df = pd.DataFrame({
    'time_minutes': df_real['time_minutes'],
    'year': df_real['year'].astype(int),
    'month': df_real['month'].astype(int),
    'hour': df_real['hour'].astype(int),
    'weekday': df_real['weekday'].astype(int),
    'watt': df_real['watt'],
    'watt_predicted': df_pred['watt']
})

# # culculate anomaly label
# comparison_df['anomaly_label'] = (np.abs(comparison_df['watt'] - comparison_df['watt_predicted']) > 0.2).astype(int)

comparison_df['relative_error'] = np.abs(comparison_df['watt'] - comparison_df['watt_predicted']) / (np.abs(comparison_df['watt']) + 1e-6)

# set threshold
threshold = 0.2
comparison_df['anomaly_label'] = (comparison_df['relative_error'] > threshold).astype(int)
# ====================
output_compare_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\lstm_prediction_comparison_with_label.csv"
comparison_df.to_csv(output_compare_path, index=False)
print(f"saved as csv: {output_compare_path}")


plt.figure(figsize=(12, 5))
plt.plot(comparison_df['watt'].values, label='True', linewidth=1)
plt.plot(comparison_df['watt_predicted'].values, label='Predicted', linewidth=1)

anomaly_indices = comparison_df[comparison_df['anomaly_label'] == 1].index
plt.scatter(anomaly_indices, comparison_df.loc[anomaly_indices, 'watt_predicted'], 
            color='red', label='Anomaly', marker='o', s=30, zorder=5)

plt.title('Model Prediction vs True Values with Anomalies')
plt.xlabel('Time step')
plt.ylabel('Watt')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

true_labels = data_normalized_test_onemonth_modification['label'].values[time_steps - 1:time_steps - 1 + len(comparison_df)]
pred_labels = comparison_df['anomaly_label'].values

precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
accuracy = accuracy_score(true_labels, pred_labels)

print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")
print(f" Accuracy:  {accuracy:.4f}")


cm = confusion_matrix(true_labels, pred_labels)
labels = ['Normal', 'Anomaly']


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
