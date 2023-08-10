import pandas as pd
import matplotlib.pyplot as plt

# Loading the CSV file
file_path = "results.csv"
results_df = pd.read_csv(file_path)

# Displaying the updated dataframe
import seaborn as sns


# Function to extract the first part of the lineage name
def extract_first_part(lineage):
    return lineage.split('.')[0]


# Applying the extracted first part to a new column
results_df['first_part'] = results_df['Lineage'].apply(extract_first_part)

# Getting unique first parts and mapping them to colors
unique_first_parts = results_df['first_part'].unique()
color_mapping = sns.color_palette("husl", len(unique_first_parts))
color_dict = {first_part: color for first_part, color in zip(unique_first_parts, color_mapping)}


# Function to get color based on first part
def get_color(lineage):
    return color_dict[lineage]


# Applying color based on the first part of the lineage
results_df['color'] = results_df['first_part'].apply(get_color)
results_df = results_df.sort_values(by='first_part')


# Function to plot updated bar chart
# Function to plot final bar chart with black outline and mean in legend
def plot_final_bar_chart(metric, ylabel, title):
    mean_value = results_df[metric].mean()
    plt.figure(figsize=(15, 5))
    bars = plt.bar(results_df['Lineage'], results_df[metric], color=results_df['color'], edgecolor='black')
    mean_line = plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.4f}')
    plt.xlabel('Lineage')
    plt.ylabel(ylabel)
    plt.title(title)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[first_part], markersize=10, label=first_part) for first_part in unique_first_parts]
    legend_handles.append(mean_line)
    plt.legend(handles=legend_handles)
    plt.xticks(rotation=90)
    plt.show()


# Plotting updated bar charts


def plot_scatter_chart(metric, ylabel, title):
    mean_value = results_df[metric].mean()
    plt.scatter(results_df['Lineage'], results_df[metric])
    plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.4f}')
    plt.xlabel('Lineage')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()


plot_final_bar_chart('accuracy', 'Accuracy', 'Accuracy by Lineage')
plot_final_bar_chart('precision', 'Precision', 'Precision by Lineage')
plot_final_bar_chart('top3acc', 'Top-3 Accuracy', 'Top-3 Accuracy by Lineage')
plot_final_bar_chart('top5acc', 'Top-5 Accuracy', 'Top-5 Accuracy by Lineage')

# Plotting scatter plots for accuracy, precision, top-3 accuracy, and top-5 accuracy
plot_scatter_chart('accuracy', 'Accuracy', 'Accuracy by Lineage')
plot_scatter_chart('precision', 'Precision', 'Precision by Lineage')
plot_scatter_chart('top3acc', 'Top-3 Accuracy', 'Top-3 Accuracy by Lineage')
plot_scatter_chart('top5acc', 'Top-5 Accuracy', 'Top-5 Accuracy by Lineage')
