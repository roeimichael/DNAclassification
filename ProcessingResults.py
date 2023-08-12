import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# Function to extract the first part of the lineage name
def extract_first_part(lineage):
    return lineage.split('.')[0]


all_files = os.listdir()

results_files = [file for file in all_files if file.startswith("results")]

for file_name in results_files:
    results_df = pd.read_csv(file_name)
    results_df['first_part'] = results_df['Lineage'].apply(extract_first_part)
    unique_first_parts = results_df['first_part'].unique()
    color_mapping = sns.color_palette("husl", len(unique_first_parts))
    color_dict = {first_part: color for first_part, color in zip(unique_first_parts, color_mapping)}
    def get_color(lineage):
        return color_dict[lineage]

    results_df['accuracy'] = results_df['accuracy'].apply(lambda x: float(x.split('(')[1].split(',')[0]))
    results_df['color'] = results_df['first_part'].apply(get_color)
    results_df = results_df.sort_values(by='first_part')
    results_df.to_csv(file_name, index=False)
    directory_path = f'./data/pictures/{file_name.split(".")[0]}/'
    os.makedirs(directory_path, exist_ok=True)


    def plot_final_bar_chart(metric, ylabel, title):
        mean_value = results_df[metric].mean()
        plt.figure(figsize=(15, 5))
        bars = plt.bar(results_df['Lineage'], results_df[metric], color=results_df['color'], edgecolor='black')
        mean_line = plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.4f}')
        plt.xlabel('Lineage')
        plt.ylabel(ylabel)
        plt.title(title)
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[first_part], markersize=10,
                       label=first_part) for first_part in unique_first_parts]
        legend_handles.append(mean_line)
        plt.legend(handles=legend_handles)
        plt.xticks(rotation=90)
        plt.savefig(f'{directory_path}{title}.png')
        plt.close()


    plot_final_bar_chart('accuracy', 'Accuracy', 'Accuracy by Lineage')
    plot_final_bar_chart('precision', 'Precision', 'Precision by Lineage')
    plot_final_bar_chart('top3acc', 'Top-3 Accuracy', 'Top-3 Accuracy by Lineage')
    plot_final_bar_chart('top5acc', 'Top-5 Accuracy', 'Top-5 Accuracy by Lineage')
