import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("./dataset/openpowerlifting.csv")

    # 1. Select features which we will use
    selected_features = ['Age', 'Sex', 'TotalKg']
    df_selected = df[selected_features].copy()
    print('Before scaling')
    print(df_selected.head())

    # 1-1. Binaryize gender column (Male : 1, Female : 0)
    df_selected.loc[:, 'Sex'] = df_selected['Sex'].map({'M': 1, 'F': 0})

    # 2. Dealing with numeric columns (Age, TotalKg) with missing values (Substitute with mean)
    df_selected['Age'].fillna(df_selected['Age'].mean())
    df_selected['TotalKg'].fillna(df_selected['TotalKg'].mean())

    # 3. Use StandardScaler to normalize (Only age and TotalKg)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected[['Age', 'TotalKg']])
    scaled_df = pd.DataFrame(scaled_data, columns=['Age', 'TotalKg'])

    # 3-1. See scaled data
    print('\nAfter scaling')
    print(scaled_df.head())

    # 4. concat scaled_df and df_selected['Sex']
    combined_df = pd.concat([scaled_df, df_selected['Sex'].reset_index(drop=True)], axis=1)
    print('\nAfter concat')
    print(combined_df.head())

    # 5. Return data
    return combined_df

preprocess()