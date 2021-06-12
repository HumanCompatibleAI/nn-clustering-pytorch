import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()

column_names = [
    "dataset", "net_type", "cg_lambda", "is_pruned", "run_num", "n-cut",
    "n-cut_mean", "n-cut_stdev", "n-cut_percentile", "n-cut_z-score",
    "test_acc", "test_loss"
]
desired_cols = [0, 1, 2, 3, 5, 9, 10, 11]
df = pd.read_csv("./results.csv", names=column_names, usecols=desired_cols)

g = sns.catplot(x="cg_lambda",
                y="n-cut_z-score",
                data=df,
                hue="dataset",
                col="is_pruned",
                row="net_type")

plt.show()
