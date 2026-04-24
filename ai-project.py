import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('Set2')
print("Libraries imported successfully")

raw = pd.read_csv('FedCycleData071012 (2).csv')

#convert to numeruc
raw['Age'] = pd.to_numeric(raw['Age'], errors='coerce')
raw['BMI'] = pd.to_numeric(raw['BMI'], errors='coerce')
raw['LengthofMenses'] = pd.to_numeric(raw['LengthofMenses'], errors='coerce')


#sort data
raw = raw.sort_values(by=['ClientID', 'CycleNumber'])


#filling missing vals
raw['Age'] = raw.groupby('ClientID')['Age'].ffill()
raw['Age'] = raw.groupby('ClientID')['Age'].bfill()

raw['BMI'] = raw.groupby('ClientID')['BMI'].ffill()
raw['BMI'] = raw.groupby('ClientID')['BMI'].bfill()


#set previous cycle by shift

raw['prev_cycle_length'] = raw.groupby('ClientID')['LengthofCycle'].shift(1)

#seting mean values and then shift
mean_values = raw.groupby('ClientID')['LengthofCycle'].expanding().mean()
mean_values = mean_values.shift(1)
mean_values = mean_values.round(2)

#fix index
raw['mean_cycle_length'] = mean_values.reset_index(level=0, drop=True)


#setting standard dev and then shift it
std_values = raw.groupby('ClientID')['LengthofCycle'].expanding().std()
std_values = std_values.shift(1)
std_values = std_values.fillna(0)
std_values = std_values.round(2)

raw['std_cycle_length'] = std_values.reset_index(level=0, drop=True)

#rmv patients with no cycle

df_cycle = raw.dropna(subset=['prev_cycle_length', 'Age', 'BMI']).copy()


df_cycle = df_cycle[
    [
        'ClientID',
        'CycleNumber',
        'Age',
        'BMI',
        'LengthofCycle',
        'prev_cycle_length',
        'mean_cycle_length',
        'std_cycle_length',
        'LengthofMenses'
    ]
]

#renaming cols
df_cycle = df_cycle.rename(columns={
    'Age': 'age',
    'BMI': 'bmi',
    'LengthofCycle': 'cycle_length',
    'LengthofMenses': 'period_length'
})


#creating target variable
df_cycle['days_to_next_period'] = df_cycle['cycle_length']


#checking for irregular cycles
df_cycle['is_irregular'] = (
    (df_cycle['cycle_length'] < 21) |
    (df_cycle['cycle_length'] > 35)
).astype(int)


#fix indexing 
df_cycle = df_cycle.reset_index(drop=True)


#printing dataset
print("Dataset A (Cycle Tracker — REAL DATA) loaded successfully")
print("Source patients :", df_cycle['ClientID'].nunique(), "real patients")
print("Shape           :", df_cycle.shape,
      "→", df_cycle.shape[0], "rows and", df_cycle.shape[1], "columns")

print(df_cycle.head(6))

# Load PCOS dataset 
df_pcos_raw = pd.read_excel(
    'PCOS_data_without_infertility.xlsx',
    sheet_name='Full_new'
)

# Select only relevant medical features
pcos_cols = [
    'PCOS (Y/N)',
    'BMI',
    'AMH(ng/mL)',
    'LH(mIU/mL)',
    'FSH(mIU/mL)',
    'FSH/LH',
    'Cycle length(days)',
    'Follicle No. (L)',
    'Follicle No. (R)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Fast food (Y/N)',
    'Reg.Exercise(Y/N)'
]

df_pcos = df_pcos_raw[pcos_cols].copy()

#converting into numercs
df_pcos['AMH(ng/mL)'] = pd.to_numeric(df_pcos['AMH(ng/mL)'], errors='coerce')
df_pcos['FSH/LH'] = pd.to_numeric(df_pcos['FSH/LH'], errors='coerce')

# Print dataset info
print("Dataset B (PCOS Clinical Data) loaded successfully")
print("Shape:", df_pcos.shape)

print(
    "PCOS positive cases:",
    df_pcos['PCOS (Y/N)'].sum(),
    "(",
    df_pcos['PCOS (Y/N)'].mean() * 100,
    "% )"
)

df_pcos.head(4)

#cheking for missing vals in other cols
print("Dataset A: Missing Values")

missing_a = df_cycle.isnull().sum()

print(missing_a[missing_a > 0].to_string())

total_missing_a = df_cycle.isnull().sum().sum()
total_values_a = df_cycle.size

#calculating total for set A
print(
    "Total missing:",
    total_missing_a,
    "out of",
    total_values_a,
    "(",
    (total_missing_a / total_values_a) * 100,
    "% )"
)

#for B
print("\nDataset B: Missing Values")

missing_b = df_pcos.isnull().sum()

print(missing_b[missing_b > 0].to_string())

total_missing_b = df_pcos.isnull().sum().sum()
total_values_b = df_pcos.size

#total for B
print(
    "Total missing:",
    total_missing_b,
    "out of",
    total_values_b,
    "(",
    (total_missing_b / total_values_b) * 100,
    "% )"
)

#fill missing values using median in set A
num_cols_a = df_cycle.select_dtypes(include='number').columns

for col in num_cols_a:
    if df_cycle[col].isnull().sum() > 0:
        median_value = df_cycle[col].median()
        df_cycle[col].fillna(median_value, inplace=True)

#fill missing values using median in set A
num_cols_b = df_pcos.select_dtypes(include='number').columns

for col in num_cols_b:
    if df_pcos[col].isnull().sum() > 0:
        median_value = df_pcos[col].median()
        df_pcos[col].fillna(median_value, inplace=True)

#double chk if there are any other missing vals 

print("Dataset A remaining missing:", df_cycle.isnull().sum().sum())
print("Dataset B remaining missing:", df_pcos.isnull().sum().sum())
print("\nAll missing values have been filled using median.")

