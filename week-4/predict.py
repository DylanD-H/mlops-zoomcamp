

import pickle
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)
args = parser.parse_args()


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month:02d}.parquet')


target = 'duration'
y_train = df[target].values

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

diff = y_train - y_pred
std = np.std(diff)
print(f'Standard deviation: {std:.2f}')

output_file = 'output.pq'
df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred,
})

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(np.mean(y_pred))




