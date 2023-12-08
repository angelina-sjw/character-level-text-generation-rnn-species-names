import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


from data_utils import load_data


######### directory and data ############
GOOGLE_COLAB = False
if GOOGLE_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    directory = '/content/drive/My Drive/spider_project'
else:
    directory = os.getcwd() 
data_path = os.path.join(directory, 'data/Spiders.csv')
species_names = set(load_data(data_path, 'species'))

#########################################

def ngram_diversity(names, n=2):
    ngrams = set()
    total_ngrams = 0
    for name in names:
        for i in range(len(name) - n + 1):
            ngrams.add(name[i:i+n])
            total_ngrams += 1
    return len(ngrams) / total_ngrams if total_ngrams > 0 else 0

def bin_occurrences_numeric(count):
    if count == 1:
        return 1
    elif 2 <= count <= 10:
        return 2
    elif 11 <= count <= 100:
        return 3
    elif 101 <= count <= 1000:
        return 4
    else:
        return 5

print('new names')
file_path = os.path.join(directory, 'experiments/new_names_only_10x.csv')

df = pd.read_csv(file_path)
df['Length'] = df['Start String'].apply(len)
df['Success'] = ~df['Attempts'].isna()
df['Success Binary'] = df['Success'].astype(int)
df['Attempts'] = df['Attempts'] + 1
df['Added Char Count'] = df.apply(lambda row: len(row['Generated New Vocab']) - len(row['Start String']) 
                                  if pd.notna(row['Generated New Vocab']) else 0, axis=1)

start_occurrences = defaultdict(int)
for name in species_names:
    for i in range(1, len(name) + 1):
        start_occurrences[name[:i]] += 1
def get_occurrence_count(start_string):
    return start_occurrences.get(start_string, 0)
df['Occurrences'] = df['Start String'].apply(get_occurrence_count).apply(bin_occurrences_numeric)

df_non_nan = df.dropna(subset=['Attempts']).copy()

success_rate = df['Success'].mean()
average_attempts = df_non_nan['Attempts'].mean()
average_added_char_count = df_non_nan['Added Char Count'].mean()
added_strings = df_non_nan.apply(lambda row: row['Generated New Vocab'][len(row['Start String']):], axis=1)
bigram_diversity = ngram_diversity(added_strings, n=2)

print(success_rate, average_attempts, average_added_char_count, bigram_diversity)

sns.lineplot(x='Length', y='Success Binary', data=df)
plt.title('Success Rate vs. Length')
plt.ylabel('Success Rate')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'new_success_vs_length.png'))
plt.clf()

sns.lineplot(x='Length', y='Attempts', data=df)
plt.title('Average Attempts vs. Length')
plt.ylabel('Average Attempts')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'new_attempts_vs_length.png'))
plt.clf()

sns.lineplot(x='Length', y='Added Char Count', data=df_non_nan)
plt.title('Added Character Counts vs. Length')
plt.ylabel('Added Character Counts')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'new_addedchar_vs_length.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Success Binary', data=df)
plt.title('Success Rate vs. Occurrences')
plt.ylabel('Success Rate')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'new_success_vs_occur.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Attempts', data=df)
plt.title('Average Attempts vs. Occurrences')
plt.ylabel('Average Attempts')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'new_attempts_vs_occur.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Added Char Count', data=df_non_nan)
plt.title('Added Character Counts vs. Occurrences')
plt.ylabel('Added Character Counts')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'new_addedchar_vs_occur.png'))
plt.clf()

correlation_length = df['Length'].corr(df['Success Binary'])
correlation_start_count = df['Occurrences'].corr(df['Success Binary'])

correlation_data = df[['Length', 'Occurrences', 'Attempts', 'Added Char Count']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlations')
plt.savefig(os.path.join(directory, 'new_correlation.png'))
plt.clf()

print(correlation_length, correlation_start_count)

print('complete names')

file_path = os.path.join(directory, 'experiments/complete_names_10x.csv')

df = pd.read_csv(file_path)
df['Length'] = df['Start String'].apply(len)
df['Success'] = ~df['Attempts'].isna()
df['Success Binary'] = df['Success'].astype(int)
df['Attempts'] = df['Attempts'] + 1
df['Added Char Count'] = df.apply(lambda row: len(row['Matched Vocab']) - len(row['Start String']) 
                                  if pd.notna(row['Matched Vocab']) else 0, axis=1)

start_occurrences = defaultdict(int)
for name in species_names:
    for i in range(1, len(name) + 1):
        start_occurrences[name[:i]] += 1
df['Occurrences'] = df['Start String'].apply(get_occurrence_count).apply(bin_occurrences_numeric)

df_non_nan = df.dropna(subset=['Attempts']).copy()

success_rate = df['Success'].mean()
average_attempts = df_non_nan['Attempts'].mean()
average_added_char_count = df_non_nan['Added Char Count'].mean()
added_strings = df_non_nan.apply(lambda row: row['Matched Vocab'][len(row['Start String']):], axis=1)
bigram_diversity = ngram_diversity(added_strings, n=2)

print(success_rate, average_attempts, average_added_char_count, bigram_diversity)

sns.lineplot(x='Length', y='Success Binary', data=df)
plt.title('Success Rate vs. Length')
plt.ylabel('Success Rate')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'complete_success_vs_length.png'))
plt.clf()

sns.lineplot(x='Length', y='Attempts', data=df)
plt.title('Average Attempts vs. Length')
plt.ylabel('Average Attempts')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'complete_attempts_vs_length.png'))
plt.clf()

sns.lineplot(x='Length', y='Added Char Count', data=df_non_nan)
plt.title('Added Character Counts vs. Length')
plt.ylabel('Added Character Counts')
plt.xlabel('Length')
plt.savefig(os.path.join(directory, 'complete_addedchar_vs_length.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Success Binary', data=df)
plt.title('Success Rate vs. Occurrences')
plt.ylabel('Success Rate')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'complete_success_vs_occur.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Attempts', data=df)
plt.title('Average Attempts vs. Occurrences')
plt.ylabel('Average Attempts')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'complete_attempts_vs_occur.png'))
plt.clf()

sns.lineplot(x='Occurrences', y='Added Char Count', data=df_non_nan)
plt.title('Added Character Counts vs. Occurrences')
plt.ylabel('Added Character Counts')
plt.xlabel('Occurrences')
plt.savefig(os.path.join(directory, 'complete_addedchar_vs_occur.png'))
plt.clf()

correlation_length = df['Length'].corr(df['Success Binary'])
correlation_start_count = df['Occurrences'].corr(df['Success Binary'])

correlation_data = df[['Length', 'Occurrences', 'Attempts', 'Added Char Count']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlations')
plt.savefig(os.path.join(directory, 'complete_correlation.png'))

print(correlation_length, correlation_start_count)