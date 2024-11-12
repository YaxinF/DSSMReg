import csv
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Please specify output directory.')

parser.add_argument('dir1', type=str, help='The name of train data.')

args = parser.parse_args()

output_folder = args.dir1


def select(file, threshold, result_file):
    df = pd.read_csv(file, index_col=0)
    mask = df > threshold
    df2 = df[mask].stack().reset_index()
    df2.columns = ['Transcription Factor', 'Target Gene', 'Similarity']
    df2.to_csv(result_file, index=False)

def filter_transcription_factors(file_path, result_file):
    df = pd.read_csv(file_path)
    # Count the number of target genes for each transcription factor
    transcription_factor_counts = df['Transcription Factor'].value_counts()
    # number of target genes  than or equal to 50
    transcription_factors = transcription_factor_counts[transcription_factor_counts >= 50].index.tolist()
    filtered_df = df[df['Transcription Factor'].isin(transcription_factors)]
    filtered_df.to_csv(result_file, index=False)

file = 'cosine_similarity_matrix.csv'
score_high = 'score_0.9.csv'
select(file, 0.9, score_high)

file2 = 'score_0.9.csv'
result_file = '50.csv'

filter_transcription_factors(file2, result_file)

# Generate  regulon file
input_file = r"50.csv"  # Replace with the actual input CSV file path
# Create a directory to store the output files
os.makedirs("output_folder", exist_ok=True)
with open(input_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    values_dict = {}

    for row in csvreader:
        value = row[0]
        second_column_value = row[1]

        if value not in values_dict:
            values_dict[value] = []

        values_dict[value].append(second_column_value)

for value, second_column_values in values_dict.items():

    output_file = os.path.join(output_folder, f"{value}.txt")

    with open(output_file, 'w') as outfile:
        outfile.write(f"{value}\n")
        for val in second_column_values:
            outfile.write(f"{val}\n")
print("success！")