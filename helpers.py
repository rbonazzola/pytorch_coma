import torch
from VTK.VTKMesh import VTKObject as Mesh
from cardiac_mesh import CardiacMesh
from torch.utils.data import TensorDataset, DataLoader, Subset
from subprocess import check_output
import shlex

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_cardiac_dataset(config):
    dataset = CardiacMesh(
        nTraining=config['nTraining'],
        nVal=config['nVal'],
        meshes_file=config['data_dir'],
        ids_file=config['ids_file'],
        reference_mesh=get_template_mesh(config),
        subpart=CardiacMesh.SUBPARTS_IDS[config['partition']],
        procrustes_scaling=config["procrustes_scaling"],
        procrustes_type=config["procrustes_type"]
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
    
    
def get_loader(cardiac_dataset, pData, batch_size, num_workers, shuffle=True):
    '''
    :param dataset:
    :param ids:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :return:
    '''           
    
    from copy import copy
    mesh_ids = cardiac_dataset.ids
    _pData = copy(pData)
    
    # cast tabular IDs to strings
    tab_ids = [str(int(x.item())) for x in _pData.idMat]
    tab_ids_set = set(tab_ids)
    
    # IDs in the intersection between mesh and tabular data
    intersec_ids = [x for x in cardiac_dataset.ids if x in tab_ids_set]
    
    # Indices of the above IDs in the mesh data
    intersec_mesh_indices = [cardiac_dataset.ids.index(x) for x in intersec_ids]
    
    # Indices of the above IDs in the tabular data
    intersec_tab_indices = [tab_ids.index(x) for x in intersec_ids]
    
    # Filter and reorder tabular data according to `intersec_tab_indices`
    _pData.idMat = pData.idMat[intersec_tab_indices]
    _pData.contMat = pData.contMat[intersec_tab_indices,:]
    _pData.catgMat = pData.catgMat[intersec_tab_indices,:]
    _pData.maskMat = pData.maskMat[intersec_tab_indices,:]
    
    # Filter and reorder mesh data according to `intersec_mesh_indices`
    meshes = [cardiac_dataset.vertices[i,:,:] for i in intersec_mesh_indices]
    
    dataset = TensorDataset(_pData.idMat, _pData.contMat, _pData.catgMat, _pData.maskMat, torch.Tensor(meshes))
    
    n_points = _pData.idMat.size(0)
    trainLoader = DataLoader(dataset = Subset(dataset,range(round(0.6*n_points))), batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)
    valLoader = DataLoader(dataset = Subset(dataset,range(round(0.6*n_points),round(0.8*n_points))), batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)
    testLoader = DataLoader(dataset = Subset(dataset,range(round(0.8*n_points),n_points)), batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)
    return trainLoader, valLoader, testLoader