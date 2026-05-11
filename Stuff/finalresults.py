import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("finalresults_tono_linear_stoch/results_N1200_sr0.8_lr0.94_sigma0.1.csv")

accs = df["accuracy"].values
print(f"Mean accuracy: {np.mean(accs):.4f}")
print(f"Std accuracy: {np.std(accs):.4f}")


# model = smf.ols(
#     'accuracy ~ C(plasticity) + C(input) + C(reservoir_size) + C(spectral_radius)',
#     data=df
# ).fit()
#
# anova_table = sm.stats.anova_lm(model, typ=2)

# print(anova_table)



