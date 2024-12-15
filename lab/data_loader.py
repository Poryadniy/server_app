import pandas as pd
import os
import zipfile
import requests


def download_and_extract_data(url, extract_to='data'):
    # Check if data already exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    zip_path = os.path.join(extract_to, 'jena_climate_2009_2016.csv.zip')
    if not os.path.exists(zip_path):
        print("Downloading data...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Data already downloaded.")

    # Extract the data
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print("Data extracted to", extract_to)


def load_data(data_path='data/jena_climate_2009_2016.csv'):
    # Load data into a pandas DataFrame
    if os.path.exists(data_path):
        print("Loading data...")
        df = pd.read_csv(data_path, encoding='ISO-8859-1')
        print("Data loaded successfully.")
        return df
    else:
        raise FileNotFoundError(f"Data file not found at {data_path}")


def main():
    url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
    download_and_extract_data(url)
    df = load_data()
    print("Data Summary:\n", df.head())


if __name__ == "__main__":
    main()