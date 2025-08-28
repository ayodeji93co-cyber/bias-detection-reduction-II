from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    # Split features and target
    X = df.drop('class_>50K', axis=1)
    y = df['class_>50K']
    return train_test_split(X, y, test_size=0.3, random_state=42)