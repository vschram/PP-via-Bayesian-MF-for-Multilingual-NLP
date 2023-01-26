from pylab import *
import pickle as pkl
import seaborn as sns


filename = f'model_PMF_task_wiki_MF_dim_10_runs_2_mo_PMF_folds_5_split_NCV_NLPerf.pkl'
with open(filename, 'rb') as t:
    results = pkl.load(t)

# Uncertainty plot over cells
fig1 = sns.heatmap(running_var[-1], xticklabels=LV['src_langs'], yticklabels=LV['tgt_langs'], cmap="YlGnBu",
                linewidth=0.5)
fig1.set(xlabel='Source', ylabel='Target')
fig1.set_title(f'Train/test split {fold + 1}')
fig1.set_xlabel("Source")
fig1.set_ylabel("Target")
plt.savefig(f'Train_Test_split_{folds + 1}.png')

plt.show()