import json
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def plot_class_distribution(df, column, title):
    class_counts = df[column].value_counts()
    class_counts.plot(kind='barh')
    plt.title(title)
    plt.xlabel('Number of Tweets')
    plt.ylabel('Gender')
    plt.show()

if __name__ == "__main__":
    # Adjust the path to your data file as necessary
    data_file = r'C:\Desktop\Narratize Data\tweet-gender-predictor\datadata.jl'
    df = load_data(data_file)

    # Assuming 'gender' is the column with gender information
    plot_class_distribution(df, 'gender', 'Frequency of Classes')