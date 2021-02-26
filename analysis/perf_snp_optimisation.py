import ExperimentClass
import numpy as np
from scipy import stats

latent_variable = "z2"
N = 1667

corrs = {}
pvals = {}

for exper in [x for x in os.listdir("output") if x.startswith("2020-11-16_13-22-45")]:
  
  print(exper)
  
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
  except:
    pass