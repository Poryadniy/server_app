import pandas as pd


def preprocess_data(df):
    # Drop any rows with missing values
    df = df.dropna()
    print("Missing values dropped.")

    # Convert the 'Date Time' column to datetime format
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    print("Date Time column converted to datetime format.")

    # Set 'Date Time' as the index
    df = df.set_index('Date Time')
    print("Date Time column set as index.")

    # Normalize the data (excluding the index)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    print("Data normalized.")

    return df_normalized


def main():
    from data_loader import load_data
    df = load_data()
    df_preprocessed = preprocess_data(df)
    print("Preprocessed Data Summary:\n", df_preprocessed.head())


if __name__ == "__main__":
    main()