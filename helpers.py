import torch
from VTK.VTKMesh import VTKObject as Mesh
from cardiac_mesh import CardiacMesh
from torch.utils.data import TensorDataset, DataLoader
from subprocess import check_output
import os
import shlex
import pickle
from Logger import logger

def get_best_gpu_device():
    '''
    This function return the GPU with the greatest amount of free memory
    '''
    import nvgpu
    gpu_info = nvgpu.gpu_info()
    free_mem = [x['mem_total']-x['mem_used'] for x in gpu_info]
    best_gpu_index = free_mem.index(max(free_mem))
    best_gpu = int(gpu_info[best_gpu_index]['index'])
    return best_gpu
    
    

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
      best_gpu = get_best_gpu_device()
      torch.cuda.set_device(best_gpu)
    return device


def scipy_to_torch_sparse(scp_matrix):
    import numpy as np
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def get_template_mesh(config):
    template_file_path = config.get('template_fname', "template/template.vtk")
    partition = config.get('partition', None)
    subpart = CardiacMesh.SUBPARTS_IDS[partition] if partition is not None else None

    template_mesh = Mesh(filename=template_file_path)
    if subpart is not None:
        template_mesh = template_mesh.extractSubpart(subpart)

    return template_mesh


def load_cardiac_dataset(config=None, mode="training"):

    if config is None:
        from config.config_parser import read_default_config
        config = read_default_config()

    if os.path.exists(config['preprocessed_data']):
      logger.info("Loading pre-aligned mesh data from {}".format(config['preprocessed_data']))
      dataset = pickle.load(open(config["preprocessed_data"], "rb"))
      if mode == "training":
          dataset.nTraining = config['nTraining']
          dataset.nVal = config['nVal']
          dataset.partition_dataset()
    else:
      dataset = CardiacMesh(
          nTraining=config['nTraining'],
          nVal=config['nVal'],
          meshes_file=config['data_dir'],
          ids_file=config['ids_file'],
          reference_mesh=get_template_mesh(config),
          subpart=CardiacMesh.SUBPARTS_IDS[config['partition']],
          procrustes_scaling=config["procrustes_scaling"],
          procrustes_type=config["procrustes_type"],
          mode=mode
      )
    
    return dataset


def get_cardiac_dataset_len(config):
    import numpy as np
    npy_file = config["data_dir"]
    with open(npy_file, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return shape

def get_current_commit_hash():
    return check_output(shlex.split("git rev-parse HEAD")).decode().strip()


def get_loader(dataset, ids, batch_size, num_workers, shuffle=True):
    '''
    :param dataset:
    :param ids:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :return:
    '''
    ids = torch.Tensor([int(x) for x in ids])
    vertices = TensorDataset(torch.Tensor(dataset), ids)
    loader = DataLoader(vertices, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
