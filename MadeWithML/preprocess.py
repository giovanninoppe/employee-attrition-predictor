from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(numerical_features, categorical_features):
    """
    Constructs a preprocessing pipeline for numerical and categorical features.

    Parameters:
    - numerical_features: list of column names containing numerical data
    - categorical_features: list of column names containing categorical data

    Returns:
    - A ColumnTransformer object that applies appropriate preprocessing to each feature type
    """

    # Pipeline for numerical features:
    # Step 1: Impute missing values using the mean of each column
    # Step 2: Scale features to have zero mean and unit variance
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for categorical features:
    # Step 1: Impute missing values using the most frequent value in each column
    # Step 2: Convert categorical variables to one-hot encoded format
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both transformers into a single ColumnTransformer
    # Applies numeric_transformer to numerical_features and categorical_transformer to categorical_features
    return ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])
