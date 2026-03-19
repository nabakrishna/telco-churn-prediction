
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_raw_data(filepath):
    return pd.read_csv(filepath)


def clean_data(df):
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned.dropna(inplace=True)
    cleaned.drop(columns=["customerID"], inplace=True)
    cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
    return cleaned


def remove_outliers_iqr(df, numeric_cols):
    cleaned = df.copy()
    for col in numeric_cols:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    return cleaned


def identify_column_types(df, target_column):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column and col != "SeniorCitizen"]
    binary_cols = ["SeniorCitizen"]
    return categorical_cols, numeric_cols, binary_cols


def build_preprocessor(categorical_cols, numeric_cols, binary_cols):
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols),
        ("bin", "passthrough", binary_cols),
    ])
    return preprocessor


def get_feature_names_after_transform(preprocessor, categorical_cols, numeric_cols, binary_cols):
    ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = ohe_feature_names + numeric_cols + binary_cols
    return all_feature_names


def split_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess_pipeline(filepath, target_column="Churn"):
    raw_df = load_raw_data(filepath)
    cleaned_df = clean_data(raw_df)

    categorical_cols, numeric_cols, binary_cols = identify_column_types(cleaned_df, target_column)
    cleaned_df = remove_outliers_iqr(cleaned_df, numeric_cols)

    X, y = split_features_and_target(cleaned_df, target_column)
    preprocessor = build_preprocessor(categorical_cols, numeric_cols, binary_cols)

    X_transformed = preprocessor.fit_transform(X)
    feature_names = get_feature_names_after_transform(preprocessor, categorical_cols, numeric_cols, binary_cols)

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    return X_transformed_df, y, preprocessor, feature_names, categorical_cols, numeric_cols, binary_cols, X