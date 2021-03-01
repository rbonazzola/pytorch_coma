import os, shlex, sys
from subprocess import check_output

os.chdir(check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii'))

sys.path.append(".")
import ExperimentClass
import numpy as np
from scipy import stats

latent_variable = "z2"
N = 1667

corrs = {}
pvals = {}

sys.path.append("data")
from get_dosages import get_dosages
kk = get_dosages("data/genotypes/rs11153730.vcf")

for exper in [x for x in os.listdir("output") if x.startswith("2020-11-16_13-22-45")]:
  
  exp_dir = os.path.join("output", exper)

  try:
    experiment = ExperimentClass.ComaExperiment(exp_dir)
    # experiment.load_model()
    experiment.load_z(drop_subset=False, inplace=True)

    z = np.array([experiment.z[latent_variable][id] for id in experiment.z.index if experiment.z.subset[id] == "test"])

    mean_dosage = np.array([int(x) for x in list(kk.values())]).mean()
    dosage = np.array([kk.get(str(id), mean_dosage) for id in experiment.z[latent_variable].index if experiment.z.subset[id] == "test"])

    # corrs = []; pvals = []
    
    # for M in range(0, 30000, N):
    #  corrs.append(stats.spearmanr(z[M:M+N], dosage[M:M+N])[0])
    #  pvals.append(stats.spearmanr(z[M:M+N], dosage[M:M+N])[1])

    corr = stats.spearmanr(z, dosage)
    corrs[exper] = corr[0]
    pvals[exper] = corr[1]
  except Exception as e:
    print(e)
    pass
 
import pandas as pd

'''
This script creates a file called `all_runs_parameters.csv` with containing the CoMA parameters for each experiment that has been performed
'''

def to_df_row(experiment):
    config = {}
    try:        
        run_id = experiment.config["run_id"]
        for k, v in experiment.config.items():
            config[k] = v if not isinstance(v, list) else ", ".join([str(x) for x in v])
        # config[] =
        train = pd.DataFrame(config, index=[run_id])
    except:
        pass
    return train

output_folder = "output"

df = pd.DataFrame()
for exper in [x for x in os.listdir(output_folder) if x.startswith("2020-11-16_13-22-45")]:
  try:
    experiment = ExperimentClass.ComaExperiment(os.path.join(output_folder, exper))
  except Exception as e:
    print(e)
    continue
    
  experiment.config["run_id"] = exper
  row = to_df_row(experiment)
  df = df.append(row)

df = df[["snp_weight", "kld_weight", "batch_size", "learning_rate", "seed"]].sort_values(axis=0, by=["kld_weight", "batch_size", "learning_rate"])
df["pvalue"] = pd.Series(pvals)
df = df[["pvalue", "snp_weight", "kld_weight", "batch_size", "learning_rate", "seed"]]

df.to_csv("pvalues.tsv", sep="\t", index=False, float_format='%.3e')
