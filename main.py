import os
import torch
import pandas as pd
import torch.nn.functional as F
from Logger import logger, timestamp
from config_parser import read_config
from model import Coma
import mesh_operations
from helpers import *
import json


__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

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


def load_pretrained_model(config, checkpoint="last"):
    '''
    This function returns a model associated to a configuration file.
    :param config: the configuration of the run.
    :param checkpoint: if "last", the last checkpoint will be selected. If "best", the best checkpoint in terms of validation loss will be selected. If an integer, the corresponding checkpoint will be selected.
    :return: the model object
    '''
    raise NotImplementedError


def main(config):

    #TODO: Print the commit hashes of all the repositories used (this, VTK, psbody)
    #TODO: Print the library versions
    #TODO: Print the hashes of the data files

    logger.info('Current Git commit hash for repository pytorch_coma: %s' % get_current_commit_hash())


    checkpoint_dir = config['checkpoint_dir']
    format_tokens = {"TIMESTAMP": timestamp}
    # TODO Put the format_tokens item into the configuration file
    checkpoint_dir = checkpoint_dir.format(**format_tokens)

    output_dir = config['output_dir']

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import logging
    log_file = config.get("log_file", None)
    if log_file is None:
        log_file = "output/%s/log" % timestamp
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Initializing parameters')
    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    # val_losses, accs, durations = [], [], []

    device = get_device()

    logger.info('Loading template mesh from %s' % config['template_fname'])
    template_mesh = get_template_mesh(config) # Used to extract the connectivity between vertices

    logger.info('Generating transform matrices')
    # - A: adjacency matrices
    # - D: downsampling matrices
    # - U: upsampling matrices
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    logger.info('Loading dataset')
    dataset = load_cardiac_dataset(config)
    logger.info("Using %s meshes for training and %s for validation." % (dataset.nTraining, dataset.nVal))
    train_loader = get_loader(dataset.vertices_train, dataset.train_ids, batch_size=batch_size, num_workers=workers_thread, shuffle=True)
    val_loader = get_loader(dataset.vertices_val, dataset.val_ids, batch_size=1, num_workers=workers_thread, shuffle=False)
    test_loader = get_loader(dataset.vertices_test, dataset.test_ids, batch_size=1, num_workers=workers_thread, shuffle=False)

    logger.info('Loading CoMA model')
    coma = Coma(config, D_t, U_t, A_t, num_nodes)

    #TODO: Use a dictionary to map the configuration parameters to this behaviour
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
    else:
        start_epoch = 1

    coma.to(device)

    if eval_flag:
        val_loss = evaluate(coma, val_loader, device)
        logger.info('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):

        logger.info("Training for epoch %s" % epoch)
        train_loss, recon_loss, kld_loss = train(coma, train_loader, optimizer, device)

        val_loss = evaluate(coma, val_loader, device)
        logger.info('  Train loss %s (%s + %s), Val loss %s' % (train_loss, recon_loss, kld_loss, val_loss))

        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        # val_losses.append(best_val_loss)

        if opt == 'sgd':
            adjust_learning_rate(optimizer, lr_decay)

    logging.info("Training finished after %s epochs" % total_epochs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Change this to use the Experiment class.
    coma.eval()
    z_file = "output/%s/latent_space.csv" % timestamp
    perf_file = "output/%s/performance.csv" % timestamp
    z_columns = ["z" + str(i) for i in range(coma.z)]
    z_df = pd.DataFrame(None, columns=z_columns)
    perf_df = pd.DataFrame(None, columns=["mse"])

    subsets = ["train", "validation", "test"]
    all_ids = []
    all_subsets = []

    logging.info("Calculating latent representation (z) and reconstruction performance measures.")

    # for i, loader in enumerate([train_loader, val_loader]):
    for i, loader in enumerate([train_loader, val_loader, test_loader]):
        subset = subsets[i]
        all_subsets += [subset] * len(loader.dataset)
        for batch, ids in loader:
            # TODO: We are duplicating code here. The best would be to change encoder according to is_variational
            batch = batch.to(device)
            if coma.is_variational:
                mu, log_var = coma.encoder(x=batch)
                z = mu # coma.sampling(mu, log_var)
            else:
                z = coma.encoder(x=batch)

            z_df = z_df.append(pd.DataFrame(data=z.cpu().detach().numpy(), columns=z_columns))
            all_ids.extend([str(int(x)) for x in ids.numpy()])

            x = coma.decoder(z)

            mse = ((x - batch) ** 2).mean(axis=1).mean(axis=1).cpu().detach().numpy()

            # batch = batch[torch.randperm(len(batch)), :, :]
            # mse_shuffled = ((x - batch) ** 2).mean(axis=1).mean(axis=1).cpu().detach().numpy()
            perf_df = perf_df.append(pd.DataFrame(data=mse, columns=["mse"]))

    perf_df['ID'] = all_ids
    perf_df = perf_df.set_index('ID')
    perf_df['subset'] = all_subsets

    z_df['ID'] = all_ids
    z_df = z_df.set_index('ID')
    z_df['subset'] = all_subsets

    z_df.to_csv(z_file)
    perf_df.to_csv(perf_file)

    with open("output/%s/config.json" % timestamp, 'w') as fp:
        config["run_id"] = timestamp
        json.dump(config, fp)

    logging.info("Execution finished")

    # This a temporary (i.e. permanent) workaround
    from subprocess import call
    import shlex
    call(shlex.split("touch output/%s/.finished" % timestamp))


def train(coma, train_loader, optimizer, device):
    coma.train()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    for i, (data, ids) in enumerate(train_loader):
        data = data.to(device)
        batch_size = data.size(0)
        optimizer.zero_grad()
        out = coma(data)
        recon_loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
        total_recon_loss += batch_size * recon_loss.item()
        loss = recon_loss
        if coma.is_variational:
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kld_loss = -0.5 * torch.mean(torch.mean(1 + coma.log_var - coma.mu ** 2 - coma.log_var.exp(), dim=1), dim=0)
            loss += coma.kld_weight * kld_loss
            total_kld_loss += batch_size * coma.kld_weight * kld_loss.item()
            # print("L1 loss = %s, DKL = %s, Total loss = %s" % (l1_loss.item(), kld_loss.item(), loss.item()))
        total_loss += batch_size * loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader.dataset), \
           total_recon_loss / len(train_loader.dataset), \
           total_kld_loss / len(train_loader.dataset)


def evaluate(coma, test_loader, device):
    coma.eval()
    total_loss = 0
    for i, (data, ids) in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            if coma.is_variational:
                mu, log_var = coma.encoder(x=data)
                z = mu # coma.sampling(mu, log_var)
            else:
                z = coma.encoder(x=data)
            out = coma.decoder(z)
        loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
        total_loss += data.size(0) * loss.item()
    return total_loss/len(test_loader.dataset)



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
