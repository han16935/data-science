import pandas as pd

df = pd.read_csv("./dataset/openpowerlifting.csv")

print('Each features\' meta data')
print(df.info())
print()

numeric_columns = df.select_dtypes(include=['int64', 'float64'])
non_numeric_columns = df.select_dtypes(include=['object'])

print('Numeric columns\' mean')
for col in numeric_columns:
    print(f'{col}\'s avg : {numeric_columns[col].mean()}')
print()

print('Numeric columns\' max')
for col in numeric_columns:
    print(f'{col}\'s avg : {numeric_columns[col].max()}')
print()

print('Numeric columns\' min')
for col in numeric_columns:
    print(f'{col}\'s avg : {numeric_columns[col].min()}')
print()

print('Numeric columns\' dirty data (For example, age cannot be less than 0)')
for col in numeric_columns:
    if col in ['Squat4Kg', 'Bench4Kg', 'Deadlift4Kg']:
        negative_cnt = (numeric_columns[col] < 0).sum()
        print(f'{col}\'s dirty data : {negative_cnt}')

    else:
        negative_cnt = (numeric_columns[col] < 0).sum()
        nan_cnt = (numeric_columns[col].isna()).sum()
        print(f'{col}\'s dirty data : {negative_cnt + nan_cnt}')
print()

print('Non-Numeric columns\' dirty data (For example, Name cannot be empty)')
for col in non_numeric_columns:
    print(f'{col}\'s dirty data : {non_numeric_columns[col].isna().sum()}')
print()

import matplotlib.pyplot as plt

for col in numeric_columns:
    plt.hist(numeric_columns[col])
    plt.title(f'{col} distribution')
    plt.xticks(rotation=360)
    plt.show()

# Since dataset is too large to see non_numeric_columns, we will plot histogram based on first 1000 data
# We will plot distribution of Sex, equipment, Weight, place
for col in non_numeric_columns:
    if col != "Name" and col != "Division":
        non_numeric_columns.head(1000)[col].value_counts().plot(kind='bar')
        plt.title(f'{col} distribution')
        plt.xticks(rotation=360)
        plt.show()

# Since there are too many unique divisions in Division column, we will plot distribution of highest 5 divisions
division_counts = non_numeric_columns['Division'].value_counts()
top_5_divisions = division_counts.head(5)

top_5_divisions.plot(kind='bar')
plt.title('Top 5 Divisions')
plt.xticks(rotation=360)
plt.show()