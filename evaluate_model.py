#TODO: TO BE FINISHED

import shlex
import os
from subprocess import check_output
import sys
import torch

os.chdir(check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii'))
from config.config_parser import read_config
from helpers import *

import ExperimentClass
exp = ExperimentClass.ComaExperiment("output/checkpoints/2020-08-20_14-36-37/")
# exp.load_model()

dataset = load_cardiac_dataset(config)
dataloader = get_loader(dataset.all_vertices, batch_size=1, num_workers=config['workers_thread'])


# checkpoint = torch.load("output/checkpoints/2020-08-18_16-34-10/checkpoint_300.pt")
# state_dict = checkpoint.get('state_dict')
# config_file = "config_files/default.cfg"
# config = read_config(config_file)

# import mesh_operations
# device = get_device()
# template_mesh = get_template_mesh(config)
# M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
# D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
# U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
# A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
# num_nodes = [len(M[i].v) for i in range(len(M))]

# from model import Coma
# coma = Coma(config, D_t, U_t, A_t, num_nodes)
# coma.load_state_dict(state_dict)
# coma.to(device)
