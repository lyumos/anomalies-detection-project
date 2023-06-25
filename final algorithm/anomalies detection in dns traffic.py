from data_for_algorithm import full_data, dc_model, nc_model, required_features, actual_labels, algorithm_results
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def data_normalization(data, features):
    required_data = data[features]
    scaler = MinMaxScaler()
    required_data_norm = pd.DataFrame(scaler.fit_transform(required_data), columns=required_data.columns)
    return required_data_norm


def data_handler(data_norm):
    dc = data_norm.columns[:7]
    nc = data_norm.columns[-7:]
    dc_df = data_norm[dc]
    nc_df = data_norm[nc]
    return dc_df, nc_df


def data_classification(predictable_data, model):
    prediction = model.predict(predictable_data)
    is_anomaly = bool(prediction)
    return is_anomaly


def res_handler(domain_anomaly, flow_anomaly, data, row_index, results_list):
    result = domain_anomaly or flow_anomaly
    results_list.append(int(result))
    if result:
        anomaly_info = ','.join(data.loc[row_index].astype(str)) + '\n'
        with open('../../anomalies-detection-project/final algorithm/anomalies_info.csv', mode='a') as file:
            if row_index == 0:
                file.write(','.join(data.columns) + '\n')
            file.write(anomaly_info)
    if row_index == len(data):
        return results_list


def quality_check(predicted, actual):
    predicted_np = np.array(predicted)
    actual_np = np.array(actual.values)
    print('Значения метрик для итогового алгоритма:\n')
    print(classification_report(actual_np, predicted_np, digits=4,
                                labels=[0, 1], target_names=['normal', 'abnormal']))


def main_logic(full_data, required_features, algorithm_results, actual_labels):
    normalized_data = data_normalization(full_data, required_features)
    for index, row in normalized_data.iterrows():
        row_df = pd.DataFrame([row.values], columns=normalized_data.columns)
        dc_df, nc_df = data_handler(row_df)
        dc_res = data_classification(dc_df, dc_model)
        nc_res = data_classification(nc_df, nc_model)
        res_handler(dc_res, nc_res, full_data, index, algorithm_results)
    quality_check(algorithm_results, actual_labels)


if __name__ == '__main__':
    main_logic(full_data, required_features, algorithm_results, actual_labels)
