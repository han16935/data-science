import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    original_df = pd.read_csv("./dataset/openpowerlifting.csv")

    # 1. Select features which we will use
    selected_features = ['Age', 'Sex', 'TotalKg']
    df_selected = df[selected_features].copy()
    print('Before encoding categorical data')
    print(df_selected.head(5))

    # 1-1. Binaryize gender column (Male : 1, Female : 0)
    df_selected.loc[:, 'Sex'] = df_selected['Sex'].map({'M': 1, 'F': 0})

    print('\nAfter encoding categorical data')
    print(df_selected.head(5))

    print('\nBefore filling NaN value')
    print(df_selected.head(5))

    # 2. Dealing with numeric columns (Age, TotalKg) with missing values (Substitute with mean)
    df_selected['Age'] = df_selected['Age'].fillna(original_df['Age'].mean())
    df_selected['TotalKg'] = df_selected['TotalKg'].fillna(original_df['TotalKg'].mean())

    print('\nAfter filling NaN value with mean')
    print(df_selected.head(5))

    print('\nBefore scaling')
    print(df_selected.head(5))

    # 3. Use StandardScaler to normalize (Only age and TotalKg)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected[['Age', 'TotalKg']])
    scaled_df = pd.DataFrame(scaled_data, columns=['Age', 'TotalKg'])

    # 3-1. See scaled data
    print('\nAfter scaling')
    print(scaled_df.head(5))

    print('\nBefore concat')
    print(scaled_df.head(5))

    # 4. concat scaled_df and df_selected['Sex']
    combined_df = pd.concat([scaled_df, df_selected['Sex'].reset_index(drop=True)], axis=1)
    print('\nAfter concat')
    print(combined_df.head(5))

    # 5. Return data
    return combined_df

df = pd.read_csv("./dataset/openpowerlifting.csv")
df_index = df['TotalKg'].isna()
combined_df = pd.concat([df.head(3), df[df_index].reset_index(drop=True)], axis=0)
preprocess(combined_df)