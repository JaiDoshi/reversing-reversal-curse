import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#-----------------

experiments = {'Vanilla':'p2d_loss.json', 'Exp1_A':'p2d_loss.json', 'Exp1_B':'p2d_loss.json', 'Exp1_C':'p2d_loss.json'}
experiments_constantNames = {'Vanilla':'p2d_loss.json', 'Exp1_A':'p2d_loss_constantNames.json', 'Exp1_B':'p2d_loss_constantNames.json', 'Exp1_C':'p2d_loss_constantNames.json'}

#-----------------

def graphFunc(df, title='Experiment 2', save='loss.png'):

    sns.set_style("whitegrid")

    # Create a figure and an axis
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot both 'LLM Loss' and 'Loss' lines
    sns.lineplot(data=df, x='Experiment', y='LLM Loss', marker='o', label='LLM Loss', color='blue', ax=ax)
    sns.lineplot(data=df, x='Experiment', y='Loss', marker='o', label='Loss', color='red', ax=ax)

    # Fill the area between the lines
    plt.fill_between(df['Experiment'], df['LLM Loss'], df['Loss'], where=(df['LLM Loss'] >= df['Loss']),
                    interpolate=True, color='green', alpha=0.3, label='LLM Loss > Loss')
    plt.fill_between(df['Experiment'], df['LLM Loss'], df['Loss'], where=(df['LLM Loss'] < df['Loss']),
                    interpolate=True, color='red', alpha=0.3, label='LLM Loss < Loss')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('Experiment')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.savefig(f'data/nlu_experiments/Exp2/{save}', dpi=300)

#-----------------

data = []
for experiment, file in experiments.items():
    expLoss = json.load(open(f'data/nlu_experiments/Exp2/{experiment}/{file}', 'r'))

    llm_loss = [prompt['llm_loss'] for prompt in expLoss]
    llm_loss_avg = sum(llm_loss)/len(llm_loss)

    loss = [prompt['loss'] for prompt in expLoss]
    loss_avg = sum(loss)/len(loss)

    data.append([experiment, llm_loss_avg, loss_avg])

df = pd.DataFrame(data, columns=['Experiment', 'LLM Loss', 'Loss'])
print(df)
df.to_csv('data/nlu_experiments/Exp2/loss.csv', index=False)
graphFunc(df, title='Experiment 2', save='loss.png')

#-----------------

data_constantNames = []
for experiment, file in experiments_constantNames.items():
    expLoss = json.load(open(f'data/nlu_experiments/Exp2/{experiment}/{file}', 'r'))

    llm_loss = [prompt['llm_loss'] for prompt in expLoss]
    llm_loss_avg = sum(llm_loss)/len(llm_loss)

    loss = [prompt['loss'] for prompt in expLoss]
    loss_avg = sum(loss)/len(loss)

    data_constantNames.append([experiment, llm_loss_avg, loss_avg])

df_constantNames = pd.DataFrame(data_constantNames, columns=['Experiment', 'LLM Loss', 'Loss'])
print(df_constantNames)
df_constantNames.to_csv('data/nlu_experiments/Exp2/loss_constantNames.csv', index=False)
graphFunc(df_constantNames, title='Experiment 2 (Constant Names)', save='loss_constantNames.png')
