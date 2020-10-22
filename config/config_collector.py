import os, shlex
from subprocess import check_output

repo_rootdir = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
os.chdir(repo_rootdir)
import sys; sys.path.append(".")
import ExperimentClass as exp
import pandas as pd

def to_df_row(experiment):
    try:
        config = {}
        run_id = experiment.config["run_id"]
        for k,v in experiment.config.items():
            config[k] = v if not isinstance(v, list) else ", ".join([str(x) for x in v])
        config[] =
        train = pd.DataFrame(config, index=[run_id])
    except:
        pass
    return train


def main(args):

    df = pd.DataFrame()
    for experiment in os.listdir(args.output_folder):
        try:
            print(experiment)
            experiment = exp.ComaExperiment(os.path.join(args.output_folder,experiment))
        except:
            continue
        row = to_df_row(experiment)
        df = df.append(row)

    df.to_csv(os.path.join(args.output_folder, "all_run_parameters.tsv"), index_label="experiment", sep="\t")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", default="output", help="Folder containing subfolders, one for each experiment performed.")
    args = parser.parse_args()
    main(args)