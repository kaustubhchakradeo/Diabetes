import pandas as pd

df = pd.read_csv('diabetes.csv')
df['BloodPressure'].replace(to_replace= [0],value='NaN', inplace=True)
df['SkinThickness'].replace(to_replace= [0],value='NaN', inplace=True)
df['Glucose'].replace(to_replace= [0],value='NaN', inplace=True)
df['BMI'].replace(to_replace= [0],value='NaN', inplace=True)


df = df.drop(df[df['BloodPressure']=='NaN'].index)
df = df.drop(df[df['BMI']=='NaN'].index)
df = df.drop(df[df['Glucose']=='NaN'].index)
df = df.drop(df[df['SkinThickness']=='NaN'].index)

print(df)

df.to_csv(r'cleaned.csv')