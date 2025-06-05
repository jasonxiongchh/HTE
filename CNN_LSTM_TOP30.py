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

import pickle
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score,mean_absolute_error, accuracy_score,confusion_matrix
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

    # 按天分组插入异常
    df_modification['date'] = df_modification['time'].dt.date
    unique_dates = df_modification['date'].unique()

    for day in unique_dates:
        daily_data = df_modification[df_modification['date'] == day]
        if daily_data.empty:
            continue

        count = 0
        while count < 100:
            # 随机选取该天的某个时间点
            random_time = np.random.choice(daily_data['time'].values)
            mask = (df_modification['time'] >= random_time) & (df_modification['time'] < (pd.to_datetime(random_time) + pd.Timedelta(minutes=1)))

            if mask.any():
                random_index = df_modification[mask].sample(n=1, random_state=42).index[0]
                original_value = df_modification.loc[random_index, 'watt']
                random_value = original_value * np.random.uniform(10, 20)

                # 修改 watt 和 label（异常标记）
                df_modification.loc[random_index, 'watt'] = random_value
                df_modification.loc[random_index, 'label'] = 1
                count += 1

    # 删除临时列
    df_modification = df_modification.drop(columns=['date'])

    # 保存修改后的文件
    output_dir = base_path  # 可以根据实际需要替换
    output_path = os.path.join(output_dir, "Electricity_HTE_testData_oneday_modification.csv")
    df_modification.to_csv(output_path, index=False)


    base_path_2 = 'F:\\uqstudy\\project\\code\\electricity_usage_monitoring\\test\\HTE_data\\final'
    file_name_2 = 'Electricity_HTE_testData_onemonth.csv' 
    modi_path_2 = os.path.join(base_path_2, file_name_2)
    df_modi_2 = pd.read_csv(modi_path_2)
    df_modi_2['time'] = pd.to_datetime(df_modi_2['time'])

    df_modification_2 = df_modi_2.copy()
    df_modification_2 = df_modification_2.sort_values(by="time").reset_index(drop=True)

    # 按天分组插入异常
    df_modification_2['date'] = df_modification_2['time'].dt.date
    unique_dates_2 = df_modification_2['date'].unique()

    for day_2 in unique_dates_2:
        daily_data_2 = df_modification_2[df_modification_2['date'] == day_2]
        if daily_data_2.empty:
            continue

        count_2 = 0
        while count_2 < 100:
            # 随机选取该天的某个时间点
            random_time_2 = np.random.choice(daily_data_2['time'].values)
            mask_2 = (df_modification_2['time'] >= random_time_2) & (df_modification_2['time'] < (pd.to_datetime(random_time_2) + pd.Timedelta(minutes=1)))

            if mask_2.any():
                random_index_2 = df_modification_2[mask_2].sample(n=1, random_state=42).index[0]
                original_value_2 = df_modification_2.loc[random_index_2, 'watt']
                random_value_2 = original_value_2 * np.random.uniform(10, 20)

                # 修改 watt 和 label（异常标记）
                df_modification_2.loc[random_index_2, 'watt'] = random_value_2
                df_modification_2.loc[random_index_2, 'label'] = 1
                count_2 += 1

    # 删除临时列
    df_modification_2 = df_modification_2.drop(columns=['date'])

    # 保存修改后的文件
    output_dir_2 = base_path_2  # 可以根据实际需要替换
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



file_path_train_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_trainData_normalized.csv"
file_path_test_normalized = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\final\Electricity_HTE_testData_normalized.csv"
file_path_test_normalized_onemonth_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\HTE_data\Electricity_HTE_testData_onemonth_modification_normalized_20.csv"

train_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_trainData_normalized_reshaped_20_all_combo.csv"
test_save_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_testData_normalized_reshaped_20_combo.csv"
test_save_path_onemonth_modification = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_testData_onemonth_modification_normalized_20_combo.csv"


time_steps = 20
features = ['watt', 'Temp','time_minutes', 'year', 'month', 'hour', 'weekday', 'season', 'Weather_numeric','label']
max_comb_size = 4


def generate_cross_features(df, features, max_comb_size, must_include='watt'):
    final_features = features.copy()
    for r in range(2, max_comb_size + 1):
        for combo in itertools.combinations(features, r):
            if must_include not in combo:
                continue
            new_feature_name = '_'.join(combo)
            df[new_feature_name] = df[list(combo)].prod(axis=1)
            final_features.append(new_feature_name)
    return df, final_features
# --------------------------- reshape ----------------------------
def reshape_data(data, time_steps, features_count):
    if len(data) < time_steps:
        raise ValueError(f"Data samples ({len(data)}) are less than the required time steps ({time_steps}).")
    reshaped_data = np.zeros((len(data) - time_steps + 1, time_steps, features_count), dtype=np.float16)
    for i in range(len(data) - time_steps + 1):
        reshaped_data[i] = data[i:i + time_steps]
    return reshaped_data

# --------------------------- save ----------------------------
def save_reshaped_data(reshaped_data, pfeatures, time_steps, save_path):
    samples = reshaped_data.shape[0]
    flattened_data = reshaped_data.reshape(samples, -1)
    column_names = [f"{feature}_t{t}" for t in range(time_steps) for feature in pfeatures]
    reshaped_df = pd.DataFrame(flattened_data, columns=column_names)
    reshaped_df.to_csv(save_path, index=False)
    print(f"Saved reshaped data to {save_path}")

def load_reshaped_data(load_path, pfeatures, time_steps):
    reshaped_df = pd.read_csv(load_path, dtype=np.float16)        
    samples = reshaped_df.shape[0]
    num_features = len(pfeatures)
    reshaped_data = reshaped_df.values.reshape((samples, time_steps, num_features))
    return reshaped_data

# --------------------------- main ----------------------------
isDataChanged = False
if isDataChanged:
    data_normalized_train = pd.read_csv(file_path_train_normalized)
    data_normalized_test = pd.read_csv(file_path_test_normalized)
    data_normalized_test_onemonth_modification = pd.read_csv(file_path_test_normalized_onemonth_modification)

    data_normalized_train, final_features = generate_cross_features(data_normalized_train, features, max_comb_size)
    data_normalized_test, _ = generate_cross_features(data_normalized_test, features, max_comb_size)
    data_normalized_test_onemonth_modification, _ = generate_cross_features(data_normalized_test_onemonth_modification, features, max_comb_size)

    # reshape
    train_data_for_model = reshape_data(data_normalized_train[final_features].values, time_steps, len(final_features))
    test_data_for_model = reshape_data(data_normalized_test[final_features].values, time_steps, len(final_features))
    test_data_for_model_onemonth_modification = reshape_data(data_normalized_test_onemonth_modification[final_features].values, time_steps, len(final_features))

    save_reshaped_data(train_data_for_model, final_features, time_steps, train_save_path)
    save_reshaped_data(test_data_for_model, final_features, time_steps, test_save_path)
    save_reshaped_data(test_data_for_model_onemonth_modification, final_features, time_steps, test_save_path_onemonth_modification)

else:
    sample_df = pd.read_csv(train_save_path, nrows=1)  
    time_features = sample_df.columns.tolist()

    feature_set = set(col.rsplit('_t', 1)[0] for col in time_features)
    final_features = sorted(feature_set)  

    train_data_for_model = load_reshaped_data(train_save_path, final_features, time_steps)
    test_data_for_model = load_reshaped_data(test_save_path, final_features, time_steps)
    test_data_for_model_onemonth_modification = load_reshaped_data(test_save_path_onemonth_modification, final_features, time_steps)

    data_normalized_train = pd.read_csv(file_path_train_normalized)
    data_normalized_test = pd.read_csv(file_path_test_normalized)
    data_normalized_test_onemonth_modification = pd.read_csv(file_path_test_normalized_onemonth_modification)

# target
y_train = data_normalized_train['watt'].values[time_steps-1:len(train_data_for_model) + time_steps-1].reshape(-1, 1)
y_test = data_normalized_test['watt'].values[time_steps-1:len(test_data_for_model) + time_steps-1].reshape(-1, 1)


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


bload_model =  True
# training
if not bload_model:
    cnn_lstm = build_lstm_model(time_steps,len(final_features))
    cnn_lstm.summary()

    history = cnn_lstm.fit(
        train_data_for_model, y_train,
        batch_size=64,
        epochs=100,
    )

    cnn_lstm.save('cnn_lstm_IF_TS20_with_all_combo.keras')
else:
    cnn_lstm = load_model('cnn_lstm_IF_TS20_with_all_combo.keras', compile=False)
    cnn_lstm.compile(optimizer='adam', loss='mse')

test_importance= False

if test_importance:

    y_pred = cnn_lstm.predict(test_data_for_model)
    baseline_mae = mean_absolute_error(y_test, y_pred)

    feature_importance = []

    for i in range(test_data_for_model.shape[2]):
        X_permuted = test_data_for_model.copy()
        np.random.shuffle(X_permuted[:, :, i])  
        
        y_pred_permuted = cnn_lstm.predict(X_permuted)
        permuted_mae = mean_absolute_error(y_test, y_pred_permuted)

        importance_score = permuted_mae - baseline_mae
        feature_importance.append(importance_score)

    importance_df = pd.DataFrame({
        'Feature': final_features,
        'Importance_Score': feature_importance
    }).sort_values(by='Importance_Score', ascending=False)

    output_dir_a = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data"
    output_filename_a = 'feature_importance_permutation.csv'
    output_path_a = os.path.join(output_dir_a, output_filename_a)
    importance_df.to_csv(output_path_a, index=False)
    print(f"Feature importance data saved to {output_path_a}")


importance_csv_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\feature_importance_permutation.csv"

compare_result = False

if compare_result:

    result  = cnn_lstm.predict(test_data_for_model_onemonth_modification)
    result = result.astype(np.float64)
    result_actul = data_normalized_test['watt'].values[:len(data_normalized_test_onemonth_modification)]
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


top_n = 30
importance_df = pd.read_csv(importance_csv_path)
selected_features = importance_df.sort_values(by='Importance_Score', ascending=False)['Feature'].iloc[:top_n].tolist()

def generate_cross_features(df, features, max_comb_size, must_include='watt'):
    final_features = features.copy()
    base_features = df.columns.tolist()
    feature_dict = {feature: df[feature] for feature in base_features}

    for feature_name in selected_features:
        if feature_name in df.columns:
            continue

        parts = []
        temp_name = feature_name
        while temp_name:
            matched = False
            for base in sorted(base_features, key=len, reverse=True):
                if temp_name.startswith(base):
                    parts.append(base)
                    temp_name = temp_name[len(base):]
                    if temp_name.startswith('_'):
                        temp_name = temp_name[1:]
                    matched = True
                    break
            if not matched:
                print(f"❌ cant：{feature_name}")
                break

        if must_include not in parts:
            continue

        try:
            new_feature = df[parts[0]]
            for p in parts[1:]:
                new_feature = new_feature * df[p]
            df[feature_name] = new_feature
            final_features.append(feature_name)
        except KeyError as e:
            print(f"❌ miss {e}，skip {feature_name}")

    return df, final_features



# ========= 基于 top 30 permutation 特征重新训练 CNN-LSTM + 做对比分析 =========

train_save_path_top30 = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_trainData_top30_reshaped.csv"
test_save_path_top30 = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_testData_top30_reshaped.csv"
test_save_path_onemonth_top30 = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\Electricity_HTE_testData_onemonth_top30_reshaped.csv"

istopDataChanged = False

if istopDataChanged:
    # ⚙️ 重新生成交叉特征并选择 top 30
    data_normalized_train_top30, _ = generate_cross_features(data_normalized_train.copy(), features, max_comb_size)
    data_normalized_test_top30, _ = generate_cross_features(data_normalized_test.copy(), features, max_comb_size)
    data_normalized_test_onemonth_top30, _ = generate_cross_features(data_normalized_test_onemonth_modification.copy(), features, max_comb_size)

    # 🔄 reshape 输入数据
    train_data_for_model_top30 = reshape_data(data_normalized_train_top30[selected_features].values, time_steps, len(selected_features))
    test_data_for_model_top30 = reshape_data(data_normalized_test_top30[selected_features].values, time_steps, len(selected_features))
    test_data_for_model_onemonth_top30 = reshape_data(data_normalized_test_onemonth_top30[selected_features].values, time_steps, len(selected_features))

    # 保存 reshape 后的输入数据
    save_reshaped_data(train_data_for_model_top30, selected_features, time_steps, train_save_path_top30)
    save_reshaped_data(test_data_for_model_top30, selected_features, time_steps, test_save_path_top30)
    save_reshaped_data(test_data_for_model_onemonth_top30, selected_features, time_steps, test_save_path_onemonth_top30)

else:
    # 加载已保存的数据
    sample_df_top30 = pd.read_csv(train_save_path_top30, nrows=1)
    time_features_top30 = sample_df_top30.columns.tolist()
    feature_set_top30 = set(col.rsplit('_t', 1)[0] for col in time_features_top30)
    selected_features = sorted(feature_set_top30)

    train_data_for_model_top30 = load_reshaped_data(train_save_path_top30, selected_features, time_steps)
    test_data_for_model_top30 = load_reshaped_data(test_save_path_top30, selected_features, time_steps)
    test_data_for_model_onemonth_top30 = load_reshaped_data(test_save_path_onemonth_top30, selected_features, time_steps)

# 🎯 创建目标 watt 值（对齐）
y_train_top30 = data_normalized_train['watt'].values[time_steps-1:len(train_data_for_model_top30)+time_steps-1].reshape(-1, 1)
y_test_top30 = data_normalized_test['watt'].values[time_steps-1:len(test_data_for_model_top30)+time_steps-1].reshape(-1, 1)
y_test_onemonth_top30 = data_normalized_test['watt'].values[:len(data_normalized_test_onemonth_modification)]
y_test_onemonth_top30 = y_test_onemonth_top30[time_steps-1:len(test_data_for_model_onemonth_top30)+time_steps-1].reshape(-1, 1)

# 保存目标值
y_train_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\y_train_top30.csv"
y_test_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\y_test_top30.csv"
y_test_onemonth_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\y_test_onemonth_top30.csv"

np.savetxt(y_train_path, y_train_top30, delimiter=',')
np.savetxt(y_test_path, y_test_top30, delimiter=',')
np.savetxt(y_test_onemonth_path, y_test_onemonth_top30, delimiter=',')


bload_model = True
# training
if not bload_model:
# 🧠 构建并训练新模型
    cnn_lstm_top30 = build_lstm_model(time_steps, len(selected_features))
    cnn_lstm_top30.summary()

    history_top30 = cnn_lstm_top30.fit(
        train_data_for_model_top30, y_train_top30,
        batch_size=64,
        epochs=100,
    )

    # 💾 保存模型
    cnn_lstm_top30.save('cnn_lstm_IF_TS20_top30_combo.keras')
else:
    cnn_lstm_top30 = load_model('cnn_lstm_IF_TS20_top30_combo.keras', compile=False)
    cnn_lstm_top30.compile(optimizer='adam', loss='mse')


# 📈 使用 onemonth 数据集预测与对比
result_top30 = cnn_lstm_top30.predict(test_data_for_model_onemonth_top30).astype(np.float64)
actual_top30 = y_test_onemonth_top30.astype(np.float64).flatten()
pred_top30 = result_top30.flatten()

# 📊 MAE计算和可视化
mae_top30 = mean_absolute_error(actual_top30, pred_top30)
print(f"MAE with top 30 features: {mae_top30:.4f}")

# 保存预测结果
prediction_df = pd.DataFrame({
    'Actual': actual_top30,
    'Predicted': pred_top30
})
prediction_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\prediction_top30.csv"
prediction_df.to_csv(prediction_path, index=False)

# 可视化对比图
plt.figure(figsize=(10, 4))
plt.plot(actual_top30, label='True')
plt.plot(pred_top30, label='Predicted')
plt.title('Prediction vs True (Top 30 Features)')
plt.xlabel('Time Step')
plt.ylabel('Watt')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 残差阈值法异常检测 ==========
normalized_test_subset = data_normalized_test_onemonth_modification[columns_to_normalize].iloc[time_steps - 1:time_steps - 1 + len(result_top30)].copy()
predicted_normalized = normalized_test_subset.copy()
predicted_normalized['watt'] = result_top30.flatten()

real_values = scaler.inverse_transform(normalized_test_subset)
predicted_values = scaler.inverse_transform(predicted_normalized)

df_real = pd.DataFrame(real_values, columns=columns_to_normalize)
df_pred = pd.DataFrame(predicted_values, columns=columns_to_normalize)

comparison_df = pd.DataFrame({
    'time_minutes': df_real['time_minutes'],
    'year': df_real['year'].astype(int),
    'month': df_real['month'].astype(int),
    'hour': df_real['hour'].astype(int),
    'weekday': df_real['weekday'].astype(int),
    'watt': df_real['watt'],
    'watt_predicted': df_pred['watt']
})

comparison_df['relative_error'] = np.abs(comparison_df['watt'] - comparison_df['watt_predicted']) / (np.abs(comparison_df['watt']) + 1e-6)
thresh = 0.20
comparison_df['anomaly_label'] = (comparison_df['relative_error'] > thresh).astype(int)

output_compare_path = r"F:\uqstudy\project\code\electricity_usage_monitoring\test\new_f_data\cnn_lstm_prediction_comparison_with_label.csv"
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

# ========== 使用 Isolation Forest 对残差进行异常检测（融合 MSE+MAE） ==========
iso_forest_params_hybrid = {
    'n_estimators': 100,
    'contamination': 0.002,
    'random_state': 42
}

IF_is_well_trained = True
if not IF_is_well_trained:
    print("Training Isolation Forest on hybrid residuals from training set...")

    train_predictions_top30 = cnn_lstm_top30.predict(train_data_for_model_top30).flatten()
    y_train_true_top30 = y_train_top30.flatten()

    train_errors_mse = (y_train_true_top30 - train_predictions_top30) ** 2
    train_errors_mae = np.abs(y_train_true_top30 - train_predictions_top30)
    hybrid_error_train = 0.5 * train_errors_mse + 0.5 * train_errors_mae

    iso_hybrid = IsolationForest(**iso_forest_params_hybrid)
    iso_hybrid.fit(hybrid_error_train.reshape(-1, 1))

    with open("IF_model_hybrid_top30.pkl", "wb") as f:
        pickle.dump(iso_hybrid, f)
else:
    with open("IF_model_hybrid_top30.pkl", "rb") as f:
        iso_hybrid = pickle.load(f)

print("Running IF on one-month test residuals...")
predicted_top30 = cnn_lstm_top30.predict(test_data_for_model_onemonth_top30).flatten()
true_top30 = y_test_onemonth_top30.flatten()

errors_mse = (true_top30 - predicted_top30) ** 2
errors_mae = np.abs(true_top30 - predicted_top30)
hybrid_error_test = 0.5 * errors_mse + 0.5 * errors_mae

hybrid_labels = (iso_hybrid.predict(hybrid_error_test.reshape(-1, 1)) == -1).astype(int)

true_labels_top30 = data_normalized_test_onemonth_modification['label'].values[time_steps - 1:time_steps - 1 + len(predicted_top30)]

print("\n[IF based on hybrid residual error (0.5*MSE + 0.5*MAE)]")
print(f"Precision: {precision_score(true_labels_top30, hybrid_labels):.4f}")
print(f"Recall:    {recall_score(true_labels_top30, hybrid_labels):.4f}")
print(f"F1 Score:  {f1_score(true_labels_top30, hybrid_labels):.4f}")
print(f"Accuracy:  {accuracy_score(true_labels_top30, hybrid_labels):.4f}")

cm = confusion_matrix(true_labels_top30, hybrid_labels)
labels = ['Normal', 'Anomaly']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Hybrid IF on Residuals)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
