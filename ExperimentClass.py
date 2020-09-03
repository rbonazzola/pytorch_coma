import os
import torch
from helpers import *
import json
import mesh_operations
import re
import pandas as pd

class Experiment():

    def __init__(self, folder):
        self.checkpoint_folder = os.path.join(folder, "checkpoints")
        self.config = None

    def get_last_chkpt(self, file_pattern="checkpoint_{epoch}.pt"):
        # from IPython import embed; embed()
        regex = re.compile("checkpoint_(.*).pt")
        last = max([int(regex.match(x).group(1)) for x in os.listdir(self.checkpoint_folder) if regex.match(x)])
        return os.path.join(self.checkpoint_folder, "checkpoint_" + str(last) + ".pt")

    def load_last_checkpoint(self):
        raise NotImplementedError

    def load_best_checkpoint(self):
        raise NotImplementedError


class DefaultPaths():
    checkpoints_dir = "performance.csv"
    performance = "performance.csv"
    z = "latent_space.csv"
    config_file = "config.json"
    gwas_dir = "GWAS"
    gwas_fp = "GWAS/z{index}.txt"


class ComaExperiment(Experiment):
    '''
    This class encapsulates all the information for a given execution:
      - Input dataset
      - Model configuration
      - Seed
      - Timestamp
      - Output paths
    It also contains methods to read the output files after the execution.
    TO DO: implement methods to write the output files during the execution.
    '''

    def __init__(self, folder='2020-08-18_16-34-10', mode='training', **kwargs):
        super(ComaExperiment, self).__init__(folder)
        self.__run_dir = folder
        self.__checkpoints_dir = kwargs.get("checkpoints_dir", DefaultPaths.checkpoints_dir)
        self.__performance = kwargs.get("performance", DefaultPaths.performance)
        self.__z = kwargs.get("z", DefaultPaths.z)
        self.__config = kwargs.get("config_file", DefaultPaths.config_file)
        self.__gwas_dir = kwargs.get("gwas_dir", DefaultPaths.gwas_dir)
        self.__mode = mode
        self.load_config()

    def load_config(self):
        config_file = os.path.join(self.__run_dir, self.__config)
        self.config = json.load(open(config_file))
        # TO FIX
        self.config['num_conv_filters'] = self.config['num_conv_filters'][1:]

    def load_model(self):
        device = get_device()
        chkpt_file = self.get_last_chkpt()
        checkpoint = torch.load(chkpt_file, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict')
        template_mesh = get_template_mesh(self.config)
        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, self.config['downsampling_factors'])
        D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
        U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
        A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
        num_nodes = [len(M[i].v) for i in range(len(M))]

        from model import Coma
        coma = Coma(self.config, D_t, U_t, A_t, num_nodes)
        coma.load_state_dict(state_dict)
        coma.to(device)
        self.model = coma


    def load_prealigned_meshes(self):
        # To avoid applying Generalized Procrustes Analysis (GPA) again.
        #TODO: change for the current config instead of the default.
        #I am doing this as a temporary workaround, but will be deprecated soon.
        from config_parser import read_default_config
        config = read_default_config()
        prealigned_meshes = config["preprocessed_data"]
        prealigned_meshes = pickle.load(open(prealigned_meshes, "rb"))
        return prealigned_meshes


    def load_z(self):
        # raise NotImplementedError
        return pd.read_csv(os.path.join(self.__run_dir, self.__z))

    def load_perf(self):
        # raise NotImplementedError
        return pd.read_csv(os.path.join(self.__run_dir, self.__performance))

    def write_config(self):
        raise NotImplementedError

    def write_z(self):
        raise NotImplementedError
        ''' Generate latent space'''
        z = self.model.encode(data=data)
        z_columns = ["z" + str(i) for i in range(z.shape[1])]
        df = pd.DataFrame(data=z, index=ids, columns=z_columns)
        return df

    def write_perf(self):
        raise NotImplementedError

    def get_cardiac_dataset(self):
        raise NotImplementedError
        from helpers import load_cardiac_dataset
        return load_cardiac_dataset(self.config)

    def __assign_set(data):
        training_ids = [(x, "training") for x in data.train_ids]
        validation_ids = [(x, "validation") for x in data.val_ids]
        testing_ids = [(x, "testing") for x in data.test_ids]
        return dict(training_ids + testing_ids + validation_ids)

    def write_id_files(self):
        self.__train_id_file = os.path.join(self.run_dir, "training_ids.txt")
        self.__test_id_file = os.path.join(self.run_dir, "testing_ids.txt")
        self.__val_id_file = os.path.join(self.run_dir, "validation_ids.txt")

        open(self.__train_id_file, "w").write("\n".join([str(x) for x in cardiac_data.train_ids]))
        open(self.__val_id_file, "w").write("\n".join([str(x) for x in cardiac_data.val_ids]))
        open(self.__test_id_file, "w").write("\n".join([str(x) for x in cardiac_data.test_ids]))

    def __enter__(self):
        return self