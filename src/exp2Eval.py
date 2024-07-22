import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#-----------------

experiments = {'Vanilla':['p2d_loss.json', 'Vanilla'], 'Exp1_A':['p2d_loss_constantNames.json', 'Control'], 'Exp1_B':['p2d_loss_constantNames.json', 'Treatment A:\nAuxiliary Augmented'], 'Exp1_C':['p2d_loss_constantNames.json', 'Treatment B:\nFull Augmented']}

#-----------------

def graphFunc(df, title='Experiment 2', save='loss.png'):

    sns.set_style("whitegrid")

    # Create a figure and an axis
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot both 'Prior Association Loss' and 'True Name Loss' lines
    sns.lineplot(data=df, x='Experiment', y='Prior Association Loss', marker='o', label='Prior Association Loss (E.g. Christopher Nolan)', color='blue', ax=ax)
    sns.lineplot(data=df, x='Experiment', y='True Name Loss', marker='o', label='True Name Loss (E.g. Daphne Barrington)', color='red', ax=ax)

    # Fill the area between the lines
    plt.fill_between(df['Experiment'], df['Prior Association Loss'], df['True Name Loss'], where=(df['Prior Association Loss'] >= df['True Name Loss']),
                    interpolate=True, color='green', alpha=0.3, label='Prior Association Loss > True Name Loss')
    plt.fill_between(df['Experiment'], df['Prior Association Loss'], df['True Name Loss'], where=(df['Prior Association Loss'] < df['True Name Loss']),
                    interpolate=True, color='red', alpha=0.3, label='Prior Association Loss < True Name Loss')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'data/nlu_experiments/Exp2/Evaluation/{save}', dpi=300)

#-----------------

def graphFuncPct(df, title='Experiment 2', save='loss_pct.png'):

    sns.set_style("whitegrid")

    # Create a figure and an axis
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot both 'Prior Association Loss' and 'Loss' lines
    sns.lineplot(data=df, x='Experiment', y='Gap %', marker='o', label='% Deviation from Prior Association Loss', color='black', ax=ax)

    # add a dotted line at 0
    plt.axhline(y=0, color='red', linestyle='--')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Loss Deviation (%)')
    plt.legend()

    # Save the plot
    plt.savefig(f'data/nlu_experiments/Exp2/Evaluation/{save}', dpi=300)

#-----------------

# Select bottom 10 gap names
vanilla = json.load(open('data/nlu_experiments/Exp2/Vanilla/p2d_loss.json', 'r'))
bottom10 = [_['name'] for _ in sorted(vanilla, key=lambda x: x['gap'])[:10]]

def keepPrompts(promptDict):
    return [prompt for prompt in promptDict if prompt['name'] in bottom10]

#-----------------

data = []
for experiment, file in experiments.items():
    expLoss = json.load(open(f'data/nlu_experiments/Exp2/{experiment}/{file[0]}', 'r'))
    expLoss = keepPrompts(expLoss)

    llm_loss = [prompt['llm_loss'] for prompt in expLoss]
    llm_loss_avg = sum(llm_loss)/len(llm_loss)

    loss = [prompt['loss'] for prompt in expLoss]
    loss_avg = sum(loss)/len(loss)

    gap_pct = [(prompt['llm_loss'] - prompt['loss'])/prompt['llm_loss'] for prompt in expLoss]
    gap_pct_avg = sum(gap_pct)/len(gap_pct)

    data.append([file[1], llm_loss_avg, loss_avg, gap_pct_avg])

df = pd.DataFrame(data, columns=['Experiment', 'Prior Association Loss', 'True Name Loss', 'Gap %'])
df.to_csv('data/nlu_experiments/Exp2/Evaluation/loss.csv', index=False)
print(df)

graphFunc(df, title='', save='loss.png')
graphFuncPct(df, title='', save='loss_pct.png')

