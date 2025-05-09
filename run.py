import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "model_comparison_results.csv"
df = pd.read_csv(file_path)

df['Model_Type'] = df['Model'].apply(lambda x: x.split('_')[1])  # A, B, C, D
df['Fine_Tuned'] = df['Model'].apply(lambda x: 'fine_tuned' in x)

model_colors = {
    'A': 'skyblue',
    'B': 'salmon',
    'C': 'lightgreen',
    'D': 'orchid'
}

metrics_to_plot = ['accuracy', 'Indeterminant_f1', 'Negative_f1', 'Positive_f1']

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.ravel()

for i, metric in enumerate(metrics_to_plot):
    ax = axs[i]
    x = np.arange(4)
    bar_width = 0.35

    for j, (model_type, group) in enumerate(df.groupby('Model_Type')):
        base_value = group.loc[~group['Fine_Tuned'], metric].values[0]
        fine_tuned_value = group.loc[group['Fine_Tuned'], metric].values[0]
        ax.bar(j - bar_width/2, base_value, width=bar_width, label=f"{model_type} base", 
               color=model_colors[model_type], alpha=0.6)
        ax.bar(j + bar_width/2, fine_tuned_value, width=bar_width, label=f"{model_type} fine-tuned", 
               color=model_colors[model_type], alpha=1.0)

    ax.set_title(metric.replace('_', ' ').title())
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(['Model A', 'Model B', 'Model C', 'Model D'])

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='medium')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
