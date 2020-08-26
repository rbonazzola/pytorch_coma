import torch
from VTK.VTKMesh import VTKObject as Mesh
from cardiac_mesh import CardiacMesh
from torch.utils.data import TensorDataset, DataLoader

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
    return check_output(shlex.split("git rev-parse HEAD")).strip()

class MyDataset(TensorDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1 # datasets should be sorted!
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]

        return x1, x2

    def __len__(self):
        return len(self.dataset1)

def get_loader(dataset, ids, batch_size, num_workers, shuffle=True):
    vertices = MyDataset(torch.Tensor(dataset), torch.Tensor([int(x) for x in ids]) )
    loader = DataLoader(vertices, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader