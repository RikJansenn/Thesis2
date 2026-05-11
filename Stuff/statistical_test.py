import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

df = pd.read_csv("bigcvs/all_results.csv")

# sns.violinplot(x='tonotopy', y='accuracy', data=df, inner='point', scale='width')
# sns.pointplot(x='tonotopy', y='accuracy', data=df, estimator='mean', color='red')
# plt.show()


df_ip_true = df[df["ip"] == True].copy()
df_ip_false = df[df["ip"] == False].copy()
df_tono_false = df[df["tonotopy"] == False].copy()

sns.violinplot(x='input_type', y='accuracy', data=df_tono_false, inner='point', scale='width')
sns.pointplot(x='input_type', y='accuracy', data=df_tono_false, estimator='mean', color='red')
plt.show()

model2 = pg.anova(dv='accuracy', between=['ip', 'input_type'], data=df_tono_false, detailed=True)
print(round(model2, 3))

# anova = AnovaRM(data=df, depvar="accuracy")

# categorical_cols = ["ip", "tonotopy", "input_type"]
#
# for c in categorical_cols:
#     df[c] = df[c].astype("category")
#
# model = ols(
#     "accuracy ~ C(ip) * C(tonotopy) * C(input_type)",
#     data=df
# ).fit()
#
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)
