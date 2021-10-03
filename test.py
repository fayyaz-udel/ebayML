import pandas as pd

df = pd.DataFrame({'TIME':['2019-10-12 09:22:00.000-04:00','2018-05-08 15:26:00.000-05:00']})

#should be slowier
#df['hour'] = pd.to_datetime(df['TIME']).dt.hour

df['hour'] = pd.to_datetime(df['TIME']).dt.hour
print (df)
