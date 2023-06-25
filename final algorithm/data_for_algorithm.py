import pandas as pd
import joblib

full_data = pd.read_csv('../../anomalies-detection-project/datasets processing/final dataset/final_dataset.csv')
dc_model = joblib.load('../../anomalies-detection-project/ml models creation/dc_model.pkl')
nc_model = joblib.load('../../anomalies-detection-project/ml models creation/nc_model.pkl')
required_features = [
      'FQDN_count', 'subdomain_length', 'longest_word', 'sld', 'len', 'lower', 'entropy',
      'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Mean',
      'Bwd Packet Length Mean', 'Fwd Packets/s', 'Bwd Packets/s'
            ]
actual_labels = pd.read_csv('../../anomalies-detection-project/datasets processing/final dataset/label_dataset.csv')
algorithm_results = []