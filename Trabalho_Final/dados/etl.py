import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

nome = 'WTG-001_Analog'


df = pd.read_csv(f'{nome}.csv', delimiter=';')
del df['Quality']
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
df['Value'] = df['Value'].str.replace(',', '.').astype(float)
df = df[df['Name'].isin(['WTG-001_Meteorological_Measurements_VentoDir','WTG-001_Terminal1_PotAt','WTG-001_Meteorological_Measurements_VentoVel3seg'])]


nome = 'WTG-001_Discrete'
df_discrete = pd.read_csv(f'{nome}.csv', delimiter=';')
del df_discrete['Quality']
df_discrete['Timestamp'] = pd.to_datetime(df_discrete['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
df_discrete['Value'] = df_discrete['Value'].astype(float)
df_discrete = df_discrete.groupby('Timestamp').agg({'Name':'last', 'Value': 'mean'})
df_discrete = df_discrete.resample('1S').ffill()
df_discrete = df_discrete.groupby('Name').resample('5T').mean()
df_discrete['Value'] = df_discrete['Value'].apply(lambda x: x if x in [1] else np.nan)
df_discrete = df_discrete.reset_index(drop=False)

df = pd.concat([df,df_discrete], ignore_index=True)
df = pd.pivot_table(df, index='Timestamp', columns='Name', values=['Value'], aggfunc='mean')
df.columns = ['_'.join(col) for col in df.columns]
df.reset_index(inplace=True)

df.rename(columns={'Value_WTG-001_Meteorological_Measurements_VentoDir': 'VentoDir','Value_WTG-001_Meteorological_Measurements_VentoVel3seg':'VentoVel','Value_WTG-001_Terminal1_PotAt':'PotenciaAtv'}, inplace=True)
df = df.dropna(how='any')
df = df[['Timestamp','PotenciaAtv','VentoDir','VentoVel']]


table = pa.Table.from_pandas(df)

pq.write_table(table, f'dados_aerogerador.parquet')
