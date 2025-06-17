import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def drop_unique_columns(df, unique_ratio=0.90):
    """Drop columns where the number of unique values == total rows or above a certain ratio."""
    to_drop = []
    for col in df.columns:
        unique_ratio_col = df[col].nunique() / len(df)
        if unique_ratio_col > unique_ratio:
            to_drop.append((col, unique_ratio_col))
    
    if to_drop:
        print("Dropping the following columns due to high cardinality:")
        for col, ratio in to_drop:
            print(f"- {col} (unique_ratio = {ratio:.2f})")
    
    return df.loc[:, [col for col in df.columns if col not in [c[0] for c in to_drop]]]

def drop_high_nulls_and_strip_strings(df, row_thresh=0.4, col_thresh=0.4):
    """Remove columns and rows with large amounts of missing data and trim strings."""
    col_null_frac = df.isnull().mean()
    df = df.loc[:, col_null_frac <= col_thresh]

    row_null_frac = df.isnull().mean(axis=1)
    df = df.loc[row_null_frac <= row_thresh]

    return df.copy().apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

def clean_boolean_like_columns(df):
    """Convert boolean-like columns to 0/1."""
    for col in df.select_dtypes(['object']).columns:
        unique_vals = df[col].dropna().unique()
        lower_vals = [str(v).lower() for v in unique_vals]
        if set(lower_vals).issubset({'yes', 'true', 'y', '1', 'no', 'false', 'n', '0'}):
            df[col] = df[col].apply(lambda v: 1 if str(v).lower() in ['yes', 'true', 'y', '1'] else 0)
    return df

def convert_to_datetime_if_possible(df):
    """Convert convertible object columns to datetime."""
    for col in df.select_dtypes(['object']).columns:
        try:
            parsed = pd.to_datetime(df[col], errors='raise')
            df[col] = parsed
        except (ValueError, TypeError):
            continue
    return df

def convert_datetime_columns_to_epoch(df):
    """Convert datetime columns to epoch timestamp."""
    for col in df.select_dtypes(['datetime64[ns]']).columns:
        df[col] = df[col].apply(lambda x: x.timestamp()) 
    return df

def convert_to_numeric_if_possible(df):
    """Convert convertible object columns to numeric."""
    for col in df.select_dtypes(['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fill_nans_df(df):
    """Fill NaNs in numeric with median and in object with mode."""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median()) 
        else:
            if not df[col].empty:
                df[col] = df[col].fillna(df[col].mode()[0]) 
    return df

def preprocess_data(df, row_thresh=0.4, col_thresh=0.4):
    """General pipeline to clean a DataFrame safely for ML models."""
    df = drop_high_nulls_and_strip_strings(df, row_thresh, col_thresh)
    df = fill_nans_df(df)
    df = drop_unique_columns(df)
    df = clean_boolean_like_columns(df)
    # df = convert_to_datetime_if_possible(df)
    # df = convert_datetime_columns_to_epoch(df)
    # df = convert_to_numeric_if_possible(df)

    if df.isnull().sum().sum() > 0:
        raise ValueError("Data still contains NaNs.")
    return df




def visualize_data(file_or_df, label=None):
    """
    Perform a general EDA (histogram + boxplot) on a DataFrame or CSV file.
    Prints a brief summary of the data, displays histograms and boxplot of numerical columns,
    and then returns the cleaned DataFrame.

    Args:
        file_or_df (str or pd.DataFrame): CSV file path or a DataFrame directly.
        label (str, optional): Target column, if applicable. Will be kept in the cleaned data.

    Returns:
        pd.DataFrame: The cleaned DataFrame after EDA.
    """
    # Loading CSV if a file path is provided
    if isinstance(file_or_df, str):
        df = pd.read_csv(file_or_df)
    else:
        df = file_or_df.copy()

    # If label is provided and not in df, raise helpful error
    if label and label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in the DataFrame. Available columns: {list(df.columns)}")

    # Summary first
    print("➥ Summary of the DataFrame:")
    print(" shape :", df.shape)
    print(" columns :", df.columns.to_list())  
    print(" info:")
    df.info()
    print(" description:")
    print(df.describe())    

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(num_cols) == 0:
        print("⚠ No numerical columns to visualize.")
        return df
    
    # Plot histograms
    df[num_cols].hist(figsize=(12, 10), bins=30, color='#60A5FA')
    plt.suptitle('Distribution of Numerical Features')
    plt.tight_layout()
    plt.show()

    # Plot boxplot
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df[num_cols], color='#FFD580')
    plt.xticks(rotation=90)
    plt.title('Boxplot of Numerical Features')
    plt.tight_layout()
    plt.show()
    
    return df




