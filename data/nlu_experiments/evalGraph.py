import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------

def graphFunc(df, save):

    # Define custom colors
    custom_colors = ["#00C8FF", "#C070FF", "#FF6782"]

    # Set the aesthetics for the plot
    sns.set_theme(style="whitegrid")

    # Create a figure and a set of subplots
    plt.figure(figsize=(10, 6))

    # Create the bar plot
    ax = sns.barplot(x='Evaluation', y='Accuracy', hue='Experiment', data=df, palette=custom_colors)

    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height() - 1), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Add 0% data labels
    for p in ax.patches:
        if p.get_height() == 0:
            ax.annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Add some customizations
    ax.set_ylabel('Accuracy (%)')
    plt.legend()

    # Hide x-axis label
    ax.set_xlabel('')

    # Save the plot
    plt.savefig(f'data/nlu_experiments/{save}.png', dpi=300)

#--------------------------

# Data
data = {
    'Experiment': ['Control', 'Control', 'Treatment A: Auxiliary Augmented', 'Treatment A: Auxiliary Augmented', 'Treatment B: Full Augmented', 'Treatment B: Full Augmented'],
    'Evaluation': ['P2D - Same Direction', 'P2D - Reverse Direction', 'P2D - Same Direction', 'P2D - Reverse Direction', 'P2D - Same Direction', 'P2D - Reverse Direction'],
    'Accuracy': [73.3, 0.0, 80.0, 3.3, 93.3, 0.0]
}

# Create DataFrame
df = pd.DataFrame(data)
graphFunc(df, 'P2D')

#--------------------------

# Data
data = {
    'Experiment': ['Control', 'Control', 'Treatment A: Auxiliary Augmented', 'Treatment A: Auxiliary Augmented', 'Treatment B: Full Augmented', 'Treatment B: Full Augmented'],
    'Evaluation': ['D2P - Same Direction', 'D2P - Reverse Direction', 'D2P - Same Direction', 'D2P - Reverse Direction', 'D2P - Same Direction', 'D2P - Reverse Direction'],
    'Accuracy': [96.7, 0.0, 100.0, 0.0, 80.0, 0.0]
}

# Create DataFrame
df = pd.DataFrame(data)
graphFunc(df, 'D2P')