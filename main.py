import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from Logger import logger, timestamp
from VTK.VTKMesh import VTKObject as Mesh
from config_parser import read_config
from model import Coma
from cardiac_mesh import CardiacMesh
import mesh_operations


def scipy_to_torch_sparse(scp_matrix):
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    values = scp_matrix.data
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def main(config):

    logger.info('Initializing parameters')

    checkpoint_dir = config['checkpoint_dir']

    format_tokens = {"TIMESTAMP": timestamp}
    checkpoint_dir = checkpoint_dir.format(**format_tokens)

    output_dir = config['output_dir']

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('template mesh from %s' % config['template_fname'])
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)
    partition = CardiacMesh.SUBPARTS_IDS[config['partition']]
    template_mesh = template_mesh.extractSubpart(partition)

    ################################################################################
    # Generate transformation matrices:
    # - A: adjacency matrices
    # - D: downsampling matrices
    # - U: upsampling matrices

    logger.info('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    ################################################################################
    ### Load data
    logger.info('Loading dataset')
    data_dir = config['data_dir']

    dataset = CardiacMesh(
        nTraining=config['nTraining'],
        nVal=config['nVal'],
        meshes_file=config['data_dir'],
        ids_file=config['ids_file'],
        reference_mesh_file=template_file_path,
        subpart=partition,
        procrustes_scaling=config["procrustes_scaling"],
        procrustes_type=config["procrustes_type"],
    )

    logger.info("Using %s meshes for training and %s for validation." % (dataset.nTraining, dataset.nVal))

    vertices_train = TensorDataset(torch.Tensor(dataset.vertices_train))
    vertices_val = TensorDataset(torch.Tensor(dataset.vertices_val))
    
    train_loader = DataLoader(vertices_train, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    val_loader = DataLoader(vertices_val, batch_size=1, shuffle=False, num_workers=workers_thread)

    logger.info('Loading CoMA model')
    ################################################################################

    start_epoch = 1
    coma = Coma(dataset.num_features, config, D_t, U_t, A_t, num_nodes)

    ################################################################################
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    # To continue from a previous run
    checkpoint_file = config['checkpoint_file']

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.to(device)

    if eval_flag:
        val_loss = evaluate(coma, val_loader, dataset.vertices_val, device)
        logger.info('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):

        logger.info("Training for epoch %s" % epoch)

        train_loss = train(coma, train_loader, len(dataset.vertices_train), optimizer, device)
        val_loss = evaluate(coma, val_loader, dataset.vertices_val, device)
            # coma, output_dir, val_loader, dataset.vertices_val, template_mesh, device, visualize=visualize)

        logger.info('  Train loss %s, Val loss %s' % (train_loss, val_loss))

        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt == 'sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        out = coma(data)
        loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
        if coma.is_variational:
            kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            loss += coma.kld_weight * kld
        total_loss += data.size(0) * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset


def evaluate(coma, test_loader, dataset, device):
    coma.eval()
    total_loss = 0
    for i, data in enumerate(test_loader):
        data = data[0].to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
        total_loss += data.size(0) * loss.item()
    return total_loss/len(dataset)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')

    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-cp', '--checkpoint_dir', default=None, help='path where checkpoints file need to be stored')
    parser.add_argument('-od', '--output_dir', default=None, help='path where to store output')
    parser.add_argument('-id', '--data_dir', default=None, help='path where to fetch input data from')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        logger.error('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    ################################################################################
    ### Read configuration
    if not os.path.exists(args.conf):
        logger.error('Config not found' + args.conf)
    config = read_config(args.conf)

    if args.data_dir:
        config['data_dir'] = args.data_dir

    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir

    if args.output_dir:
        config['output_dir'] = args.output_dir

    main(config)
