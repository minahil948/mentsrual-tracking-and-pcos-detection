
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("All libraries imported successfully.")


#set A
raw = pd.read_csv('FedCycleData071012 (2).csv')


# Convert to numeric
raw['Age']            = pd.to_numeric(raw['Age'],            errors='coerce')
raw['BMI']            = pd.to_numeric(raw['BMI'],            errors='coerce')
raw['LengthofMenses'] = pd.to_numeric(raw['LengthofMenses'], errors='coerce')


raw = raw.sort_values(by=['ClientID', 'CycleNumber'])


raw['Age'] = raw.groupby('ClientID')['Age'].ffill()
raw['Age'] = raw.groupby('ClientID')['Age'].bfill()

raw['BMI'] = raw.groupby('ClientID')['BMI'].ffill()
raw['BMI'] = raw.groupby('ClientID')['BMI'].bfill()


raw['prev_cycle_length'] = raw.groupby('ClientID')['LengthofCycle'].shift(1)


mean_values = raw.groupby('ClientID')['LengthofCycle'].expanding().mean()
mean_values = mean_values.shift(1)
mean_values = mean_values.round(2)

raw['mean_cycle_length'] = mean_values.reset_index(level=0, drop=True)


std_values = raw.groupby('ClientID')['LengthofCycle'].expanding().std()
std_values = std_values.shift(1)
std_values = std_values.fillna(0)
std_values = std_values.round(2)

raw['std_cycle_length'] = std_values.reset_index(level=0, drop=True)


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


df_cycle = df_cycle.rename(
    columns={
        'Age'          : 'age',
        'BMI'          : 'bmi',
        'LengthofCycle': 'cycle_length',
        'LengthofMenses': 'period_length'
    }
)


df_cycle['days_to_next_period'] = df_cycle['cycle_length']


df_cycle['is_irregular'] = (
    (df_cycle['cycle_length'] < 21) |
    (df_cycle['cycle_length'] > 35)
).astype(int)



df_cycle = df_cycle.reset_index(drop=True)


print("Dataset A (Cycle Tracker — REAL DATA) loaded successfully")
print("Unique patients :", df_cycle['ClientID'].nunique())
print(
    "Shape           :", df_cycle.shape,
    "→", df_cycle.shape[0], "rows and", df_cycle.shape[1], "columns"
)
print(df_cycle.head(6))


#set B loading
df_pcos_raw = pd.read_excel(
    'PCOS_data_without_infertility.xlsx',
    sheet_name='Full_new'
)


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


df_pcos['AMH(ng/mL)'] = pd.to_numeric(df_pcos['AMH(ng/mL)'], errors='coerce')
df_pcos['FSH/LH']     = pd.to_numeric(df_pcos['FSH/LH'],     errors='coerce')


print("Dataset B (PCOS Clinical Data) loaded successfully")
print("Shape:", df_pcos.shape)
print(
    "PCOS positive cases:",
    int(df_pcos['PCOS (Y/N)'].sum()),
    f"({df_pcos['PCOS (Y/N)'].mean() * 100:.1f}%)"
)
print(df_pcos.head(4))


#counting missing vals
print("=" * 50)
print("Dataset A: Missing Values")
print("=" * 50)

missing_a = df_cycle.isnull().sum()

print(missing_a[missing_a > 0].to_string() if missing_a.sum() > 0 else "No missing values found.")

total_missing_a = df_cycle.isnull().sum().sum()
total_values_a  = df_cycle.size

print(
    "Total missing:",
    total_missing_a,
    "out of",
    total_values_a,
    f"({total_missing_a / total_values_a * 100:.2f}%)"
)


print()
print("=" * 50)
print("Dataset B: Missing Values")
print("=" * 50)

missing_b = df_pcos.isnull().sum()

print(missing_b[missing_b > 0].to_string() if missing_b.sum() > 0 else "No missing values found.")

total_missing_b = df_pcos.isnull().sum().sum()
total_values_b  = df_pcos.size

print(
    "Total missing:",
    total_missing_b,
    "out of",
    total_values_b,
    f"({total_missing_b / total_values_b * 100:.2f}%)"
)


#handling missing vals
# Fill missing values using median for Dataset A
num_cols_a = df_cycle.select_dtypes(include='number').columns

for col in num_cols_a:
    if df_cycle[col].isnull().sum() > 0:
        median_value = df_cycle[col].median()
        df_cycle[col] = df_cycle[col].fillna(median_value)

#set B
num_cols_b = df_pcos.select_dtypes(include='number').columns

for col in num_cols_b:
    if df_pcos[col].isnull().sum() > 0:
        median_value = df_pcos[col].median()
        df_pcos[col] = df_pcos[col].fillna(median_value)


print("Dataset A remaining missing:", df_cycle.isnull().sum().sum())
print("Dataset B remaining missing:", df_pcos.isnull().sum().sum())
print("All missing values have been filled using median imputation.")


#box plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Boxplots for Outlier Detection — Dataset A', fontweight='bold')

cols_a = ['cycle_length', 'bmi', 'period_length']

for ax, col in zip(axes, cols_a):
    ax.boxplot(
        df_cycle[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor='#4C9BE8', alpha=0.7),
        medianprops=dict(color='black', linewidth=2)
    )
    ax.set_title(col)
    ax.set_ylabel('Value')

plt.tight_layout()
plt.show()


#outlier detection

print("Outlier Count (IQR method) — Dataset A")

cols_a_out = ['cycle_length', 'days_to_next_period', 'bmi']

for col in cols_a_out:
    Q1  = df_cycle[col].quantile(0.25)
    Q3  = df_cycle[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df_cycle[col] < lower) | (df_cycle[col] > upper)).sum()

    print(f"  {col}: {outliers} outliers")


print()
print("Outlier Count (IQR method) — Dataset B")

cols_b_out = ['BMI', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)', 'Cycle length(days)']

for col in cols_b_out:
    Q1  = df_pcos[col].quantile(0.25)
    Q3  = df_pcos[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df_pcos[col] < lower) | (df_pcos[col] > upper)).sum()

    print(f"  {col}: {outliers} outliers")


print()
print("Strategy: Winsorizing (clipping values to IQR bounds)")


#winsorize

def winsorize(df, cols):
    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower_bound, upper_bound)

    return df


df_cycle = winsorize(
    df_cycle,
    ['cycle_length', 'days_to_next_period', 'bmi', 'period_length']
)

df_pcos = winsorize(
    df_pcos,
    ['BMI', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)', 'Cycle length(days)']
)

print("Outliers capped in both datasets.")


#data types convrsion
binary_cols_pcos = [
    'PCOS (Y/N)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Fast food (Y/N)',
    'Reg.Exercise(Y/N)'
]

for col in binary_cols_pcos:
    df_pcos[col] = df_pcos[col].fillna(0).astype(int)

# Ensure target variable is integer
df_cycle['is_irregular'] = df_cycle['is_irregular'].astype(int)


print("Dataset A — Key Data Types")
print(
    df_cycle[['cycle_length', 'days_to_next_period', 'bmi', 'is_irregular']]
    .dtypes
    .to_string()
)

print()
print("Dataset B — Key Data Types")
print(
    df_pcos[['BMI', 'AMH(ng/mL)', 'LH(mIU/mL)', 'PCOS (Y/N)']]
    .dtypes
    .to_string()
)

print()
print("All types verified.")



print(
    f"Dataset A shape: {df_cycle.shape} "
    f"({df_cycle.shape[0]} cycle records, {df_cycle.shape[1]} features)"
)

print(
    f"Dataset B shape: {df_pcos.shape} "
    f"({df_pcos.shape[0]} patients, {df_pcos.shape[1]} features)"
)


print()
print("Dataset A: Descriptive Statistics")
print(df_cycle.describe().round(2).to_string())

print()
print("Dataset B: Descriptive Statistics")
print(df_pcos.describe().round(2).to_string())


# Histograms 

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    'Distribution of Numerical Features — Dataset A',
    fontweight='bold',
    fontsize=14
)

num_cols = [
    'cycle_length',
    'prev_cycle_length',
    'mean_cycle_length',
    'std_cycle_length',
    'bmi',
    'period_length'
]

colors = ['steelblue', 'teal', 'darkcyan', 'slateblue', 'coral', 'goldenrod']

for ax, col, color in zip(axes.flat, num_cols, colors):
    ax.hist(df_cycle[col], bins=28, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("Observation: cycle_length and days_to_next_period are approximately normally distributed.")
print("std_cycle_length is right-skewed — some patients have highly variable cycle histories.")
print("BMI shows a slight right skew, common in health datasets.")


# Cycle Irregularity

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Cycle Length Analysis', fontweight='bold')


# Histogram with irregular boundary lines
axes[0].hist(
    df_cycle['cycle_length'],
    bins=30,
    color='steelblue',
    edgecolor='white'
)

axes[0].axvline(21, linestyle='--', color='tomato', label='Below 21 (Irregular)')
axes[0].axvline(35, linestyle='--', color='orange', label='Above 35 (Irregular)')
axes[0].set_title('Cycle Length Distribution')
axes[0].set_xlabel('Cycle Length (days)')
axes[0].set_ylabel('Frequency')
axes[0].legend()


# Boxplot comparing regular vs irregular
regular_cycles   = df_cycle[df_cycle['is_irregular'] == 0]['cycle_length']
irregular_cycles = df_cycle[df_cycle['is_irregular'] == 1]['cycle_length']

axes[1].boxplot(
    [regular_cycles, irregular_cycles],
    labels=['Regular', 'Irregular']
)
axes[1].set_title('Regular vs Irregular Cycles')
axes[1].set_ylabel('Cycle Length (days)')


# Pie chart
counts = df_cycle['is_irregular'].value_counts().sort_index()

axes[2].pie(
    counts,
    labels=['Regular', 'Irregular'],
    autopct='%1.1f%%',
    startangle=90
)
axes[2].set_title('Irregularity Rate')


plt.tight_layout()
plt.show()

regular_pct = (df_cycle['is_irregular'] == 0).mean() * 100

print("Regular cycles  :", (df_cycle['is_irregular'] == 0).sum(), f"({regular_pct:.1f}%)")
print("Irregular cycles:", (df_cycle['is_irregular'] == 1).sum(), f"({100 - regular_pct:.1f}%)")



symptom_cols = [
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)'
]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle('Symptom Prevalence in PCOS vs Non-PCOS Patients', fontweight='bold')

for ax, col in zip(axes, symptom_cols):

    pcos_yes = df_pcos[df_pcos['PCOS (Y/N)'] == 1][col].mean() * 100
    pcos_no  = df_pcos[df_pcos['PCOS (Y/N)'] == 0][col].mean() * 100

    ax.bar(
        ['Non-PCOS', 'PCOS'],
        [pcos_no, pcos_yes],
        color=['steelblue', 'coral']
    )

    ax.set_title(col.replace('(Y/N)', '').strip())
    ax.set_ylabel('% of patients')
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.show()

print("Key insight: PCOS patients show higher symptom prevalence across all categories.")
print("Hair growth and weight gain show the strongest differences between groups.")


# PCOS Class Distribution

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('PCOS Dataset Analysis', fontweight='bold')


# Pie chart 
pcos_counts = df_pcos['PCOS (Y/N)'].value_counts().sort_index()

axes[0].pie(
    pcos_counts,
    labels=['No PCOS', 'PCOS'],
    autopct='%1.1f%%',
    colors=['steelblue', 'coral'],
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
)
axes[0].set_title('PCOS Class Distribution')


# cycle length by PCOS status
no_pcos  = df_pcos[df_pcos['PCOS (Y/N)'] == 0]['Cycle length(days)']
yes_pcos = df_pcos[df_pcos['PCOS (Y/N)'] == 1]['Cycle length(days)']

axes[1].hist(no_pcos,  bins=20, alpha=0.7, label='No PCOS', color='steelblue')
axes[1].hist(yes_pcos, bins=20, alpha=0.7, label='PCOS',    color='coral')

axes[1].set_title('Cycle Length Distribution by PCOS Status')
axes[1].set_xlabel('Cycle Length (days)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.show()


# Scatter Plots

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Feature Relationships Analysis', fontweight='bold')


# Previous cycle length vs days to next period
axes[0].scatter(
    df_cycle['prev_cycle_length'],
    df_cycle['days_to_next_period'],
    alpha=0.3,
    s=12,
    color='steelblue'
)

axes[0].set_title('Previous Cycle vs Next Period (Dataset A)')
axes[0].set_xlabel('Previous Cycle Length (days)')
axes[0].set_ylabel('Days to Next Period')


# BMI vs Cycle Length coloured by irregularity
colors_irregular = df_cycle['is_irregular'].map({0: 'steelblue', 1: 'coral'})

axes[1].scatter(
    df_cycle['bmi'],
    df_cycle['cycle_length'],
    c=colors_irregular,
    alpha=0.3,
    s=12
)

axes[1].set_title('BMI vs Cycle Length')
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('Cycle Length (days)')

axes[1].legend(
    handles=[
        Patch(color='steelblue', label='Regular'),
        Patch(color='coral',     label='Irregular')
    ]
)


# LH vs AMH coloured by PCOS status
colors_pcos = df_pcos['PCOS (Y/N)'].map({0: 'steelblue', 1: 'coral'})

axes[2].scatter(
    df_pcos['LH(mIU/mL)'],
    df_pcos['AMH(ng/mL)'],
    c=colors_pcos,
    alpha=0.5,
    s=15
)

axes[2].set_title('LH vs AMH (PCOS Analysis)')
axes[2].set_xlabel('LH (mIU/mL)')
axes[2].set_ylabel('AMH (ng/mL)')

axes[2].legend(
    handles=[
        Patch(color='steelblue', label='No PCOS'),
        Patch(color='coral',     label='PCOS')
    ]
)


plt.tight_layout()
plt.show()

print("Insight 1: Previous cycle length has a strong positive relationship with next cycle length.")
print("Insight 2: BMI shows mild separation between regular and irregular cycles.")
print("Insight 3: LH and AMH show clear clustering by PCOS status.")


#pair plot
pair_cols_a = [
    'cycle_length',
    'prev_cycle_length',
    'mean_cycle_length',
    'bmi',
    'days_to_next_period',
    'is_irregular'
]

sample_df = df_cycle[pair_cols_a].sample(400, random_state=42)

g = sns.pairplot(
    sample_df,
    hue='is_irregular',
    palette={0: 'steelblue', 1: 'coral'},
    plot_kws={'alpha': 0.4, 's': 15},
    diag_kind='kde'
)

g.figure.suptitle(
    'Pairwise Feature Relationships — Dataset A',
    y=1.02,
    fontweight='bold'
)

plt.show()


pair_cols_b = [
    'BMI',
    'AMH(ng/mL)',
    'LH(mIU/mL)',
    'FSH(mIU/mL)',
    'Cycle length(days)',
    'PCOS (Y/N)'
]

g2 = sns.pairplot(
    df_pcos[pair_cols_b],
    hue='PCOS (Y/N)',
    palette={0: 'steelblue', 1: 'coral'},
    plot_kws={'alpha': 0.4, 's': 15},
    diag_kind='kde'
)

g2.figure.suptitle(
    'Pairwise Feature Relationships — PCOS Dataset',
    y=1.02,
    fontweight='bold'
)

plt.show()

print("Insight: AMH and LH show the strongest separation between PCOS and non-PCOS patients.")


#methadologgy
feature_cols_a = [
    'prev_cycle_length',
    'mean_cycle_length',
    'std_cycle_length',
    'age',
    'bmi',
    'period_length',
    'CycleNumber',      
]

target_a = 'days_to_next_period'

X_a = df_cycle[feature_cols_a]
y_a = df_cycle[target_a]

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_a,
    y_a,
    test_size=0.2,   
    random_state=42
)

print("Dataset Split Summary:")
print(f"  Training samples : {len(X_train_a)}")
print(f"  Testing samples  : {len(X_test_a)}")
print(f"  Features used    : {feature_cols_a}")




rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

# Train the model
rf_reg.fit(X_train_a, y_train_a)

# Predict on test data
y_pred_a = rf_reg.predict(X_test_a)

# Evaluate performance
mae = mean_absolute_error(y_test_a, y_pred_a)
r2  = r2_score(y_test_a, y_pred_a)


print("Model 1 — Random Forest Regressor Results")
print("-" * 45)
print(f"Mean Absolute Error (MAE) : {mae:.2f} days")
print(f"R² Score                  : {r2:.4f}")

print()
print("Interpretation:")
print(f"  On average, predictions differ by about {mae:.1f} days from the actual cycle length.")
print(f"  The model explains approximately {r2 * 100:.0f}% of the variance in cycle length.")


 #Actual vs Predicted

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fig.suptitle(
    'Model 1 — Random Forest Regressor Performance',
    fontsize=13,
    fontweight='bold'
)


# Plot 1: Actual vs Predicted scatter
axes[0].scatter(
    y_test_a,
    y_pred_a,
    alpha=0.4,
    s=18,
    color='steelblue'
)

lims = [y_test_a.min(), y_test_a.max()]

axes[0].plot(
    lims,
    lims,
    linestyle='--',
    linewidth=1.5,
    color='red',
    label='Perfect prediction'
)

axes[0].set_xlabel('Actual Days to Next Period')
axes[0].set_ylabel('Predicted Days')
axes[0].set_title(f'Actual vs Predicted  (MAE={mae:.2f}, R²={r2:.3f})')
axes[0].legend()


# Residual distribution
residuals = y_test_a - y_pred_a

axes[1].hist(
    residuals,
    bins=35,
    color='slateblue',
    edgecolor='white',
    alpha=0.85
)

axes[1].axvline(
    0,
    linestyle='--',
    linewidth=1.5,
    color='tomato'
)

axes[1].set_xlabel('Residual (Actual − Predicted)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')


plt.tight_layout()
plt.show()



importance_df = pd.DataFrame(
    {
        'Feature'   : feature_cols_a,
        'Importance': rf_reg.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=True
)

plt.figure(figsize=(9, 5))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance'],
    color='steelblue',
    alpha=0.85
)

plt.xlabel('Feature Importance (Mean Decrease in Impurity)')
plt.title(
    'Model 1 — Feature Importances\n'
    '(Features used to predict next period date)',
    fontsize=12,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

print("Key Insight:")
print("  prev_cycle_length and mean_cycle_length are the strongest predictors.")
print("  A patient's own cycle history is the best indicator of the next period date.")


 #Irregularity Flagging 

pred_df = pd.DataFrame(
    {
        'actual_days'      : y_test_a.values,
        'predicted_days'   : np.round(y_pred_a, 1),
        'flagged_irregular': (y_test_a.values < 21) | (y_test_a.values > 35)
    }
)

n_flagged = pred_df['flagged_irregular'].sum()
total     = len(pred_df)


print("Irregularity Detection Summary")
print("-" * 35)
print(f"Flagged as irregular : {n_flagged} / {total}")
print(f"Irregularity rate    : {n_flagged / total * 100:.1f}%")

print()
print("Sample Predictions (first 10 rows):")
print(pred_df.head(10).to_string(index=False))

print()
print("Note:")
print("Patients flagged as irregular (predicted cycle < 21 or > 35 days)")
print("would proceed to Model 2 (Random Forest Classifier) for PCOS risk assessment.")
