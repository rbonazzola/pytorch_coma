import os
import torch
import pandas as pd
import torch.nn.functional as F
from utils.logger import logger, timestamp
from config.config_parser import read_config
from model import Coma
from utils import mesh_operations
from utils.helpers import *
import json
from pprint import pprint
from copy import copy
import logging
import shutil

# This is necessary in order to be able to import the pkl file with the preprocessed cardiac data
import sys; sys.path.append("data")
from cardiac_mesh import CardiacMesh

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


def save_training_info(loss_info, filename):
    """
    :param loss_info: list of lists, containing six elements per "row" (the different losses), and one row per epoch
    :return:
    """
    #TODO: use a Pandas DataFrame for this
    column_names = ["epoch", "reconstruction_loss_training", "KLD_loss_training", "loss_training", "reconstruction_loss_eval", "KLD_loss_eval", "loss_eval"]
    with open(filename, "w") as handle:
        handle.write("%s\n" % ",".join(column_names))
        for row in loss_info:
            handle.write("%s\n" % ( ",".join(["%d"] + ["%.5f"]*(len(row)-1)) % tuple(row) ) )


def log_loss_info(losses, rec_loss_name, wkl, wsnp):
    losses = losses[0:3] + [rec_loss_name] + losses[3:]
    # from IPython import embed; embed()
    logger.info(
        ('Epoch {:d}/{:d}:\n\t' +
        'Train set: {:.5f} ({}) + ' + str(wkl) + ' * {:.5f} (KL) - ' + str(wsnp) + ' * {:.5f} (SNP) = {:.5f},\t' +
        'Validation set: {:.5f} + ' + str(wkl) + ' * {:.5f} = {:.5f}')
        .format(*losses)
    )


def main(config):

    #This function is too long!
    #TODO IMPORTANT: Use the Experiment class to modularize the different steps

    #TODO: Print the commit hashes of all the repositories used (this, VTK, psbody)
    #TODO: Print the library versions
    #TODO: Print the hashes of the data files

    if config['seed'] is not None:
      torch.manual_seed(config['seed'])
      # np.random.seed(config['seed'])
    else:
      config['seed'] = torch.seed()

    logger.info('Current Git commit hash for repository pytorch_coma: %s' % get_current_commit_hash())

    output_dir = config['output_dir']
    format_tokens = {"TIMESTAMP": timestamp}
    
    #TODO: Put the format_tokens item into the configuration file
    output_dir = output_dir.format(**format_tokens)
    config['output_dir'] = output_dir
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    log_file = config.get("log_file", None)
    if log_file is None:
        log_file = "{}/log".format(output_dir)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Initializing parameters')
    
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    
    opt = config['optimizer']
    batch_size = config['batch_size']
    total_epochs = config['epoch']
    
    workers_thread = config['workers_thread']
    device = get_device()

    is_this_a_test = config['test']
    
    logger.info('Loading template mesh from %s' % config['template_fname'])
    template_mesh = get_template_mesh(config) # Used to extract the connectivity between vertices

    logger.info('Generating transform matrices')
    # - A|D|U: adjacency|downsampling|upsampling matrices
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A, D, U))
    num_nodes = [len(M[i].v) for i in range(len(M))]

    logger.info('Loading dataset')
    dataset = load_cardiac_dataset(config)

    logger.info("Using %s meshes for training and %s for validation." % (config["nTraining"], config["nVal"]))
    train_loader, val_loader, test_loader = get_loader(dataset, config["nTraining"], config["nVal"], batch_size, workers_thread, shuffle=True)

    logger.info('Loading CoMA model')
    coma = Coma(config, dataset.num_features, D_t, U_t, A_t, num_nodes)
    
    # TODO: Add this into the model Class itself
    n_channels = [config["num_conv_filters"][0]] + [x for x in config["num_conv_filters"][1:] for _ in (0,1)]
    n_channels[-1] = 1
    n_vertices = [x for x in num_nodes for _ in (0,1)] + [config["z"]]
    architecture = list(zip(n_vertices, n_channels))
    logger.info('Decoder architecture is:\n' + " -> ".join([str(x) for x in architecture]))

    optimizers_menu = {
      "adam": torch.optim.Adam(coma.parameters(), lr=lr, betas=(0.5,0.99), weight_decay=weight_decay),
      "sgd": torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    }

    losses_menu = {
      "l1": {"name": "L1", "function": F.l1_loss},
      "mse": {"name": "MSE", "function": F.mse_loss}
    }

    try:
        optimizer = optimizers_menu[opt]
    except KeyError:
        raise Exception("No valid optimizer provided")

    try:
        rec_loss_fun = losses_menu[config["reconstruction_loss"]]["function"]
        rec_loss_name = losses_menu[config["reconstruction_loss"]]["name"]
    except KeyError:
        raise Exception("No valid reconstruction loss provided")

    # To continue from a previous run saved in checkpoint_file
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        logging.info("Resuming training from previous execution: %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        start_epoch = 1

    coma.to(device)

    with open("{}/config.json".format(output_dir), 'w') as fp:
        config["run_id"] = timestamp
        json.dump(config, fp, indent=4)

    best_val_loss = float('inf')
    
    loss_df = pd.DataFrame(columns=["epoch", "reconstruction_loss", "kld_loss", "snp_loss", "total_loss", "subset"])
    last_losses = []

    for epoch in range(start_epoch, total_epochs + 1):

        losses_train = train(coma, train_loader, rec_loss_fun, optimizer, device)
        losses_val = evaluate(coma, val_loader, rec_loss_fun, device)
        loss_df = loss_df.append(pd.DataFrame([[epoch] + list(losses_train) + ["train"]], columns=loss_df.columns))
        # loss_df = loss_df.append(pd.DataFrame([[epoch] + list(losses_val) + ["validation"]], columns=loss_df.columns))        
        
        log_loss_info([epoch, total_epochs] + list(losses_train) + list(losses_val), rec_loss_name, coma.kld_weight, 0.1)

        # To get the best run in the validation subset.
        if losses_val[-1] < best_val_loss:
            best_val_loss = losses_val[-1]
            best_epoch = epoch
            best_model = copy(coma)
            if not config['save_all_models']:0.1
                save_model(coma, optimizer, epoch, losses_train[-1], losses_val[-1], checkpoint_dir)

        if config['save_all_models']:
            save_model(coma, optimizer, epoch, losses_train[-1], losses_val[-1], checkpoint_dir)

        if config["stop_if_not_learning"]:
            if len(last_losses) > 20:                
                current_loss = loss_df.iloc[-1]["total_loss"] # last validation loss                
                loss_ref = last_losses.pop(0) # loss 20 epochs ago
                if current_loss >= loss_ref:
                    logging.info("Stopping training early at epoch {}. The network has not been learning in the last 20 epochs.".format(epoch))
                    break
            last_losses.append(loss_df.iloc[-1]["total_loss"])

        if opt == 'sgd':
            adjust_learning_rate(optimizer, lr_decay)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logging.info("Training finished after %s epochs." % epoch) 
    logging.info("The best model yields validation loss = %.5f (at epoch %s)." % (best_val_loss, str(best_epoch)))
    loss_df.to_csv("{}/training_losses.csv".format(output_dir), index=False, float_format="%.5f")

    if best_val_loss > 0.9:
        logging.error("The model did not learn. Try changing the parameters of the networks or of the training process.")
        if not is_this_a_test:
            shutil.rmtree(checkpoint_dir)
            exit()

    logging.info("Saving best model state in {}/best_model.pkl and last model state in {}/last_model.pkl".format(output_dir, output_dir))
    torch.save(best_model.state_dict(), "{}/best_model.pkl".format(output_dir))
    torch.save(coma.state_dict(), "{}/last_model.pkl".format(output_dir))
   
    # Generate files of 
    # - 1) performance (MSE),
    # - 2) latent representations of each mesh 
    #TODO: Change all this code to use the Experiment class.
    coma = best_model
    coma.eval()
    z_file = "{}/latent_space.csv".format(output_dir)
    perf_file = "{}/performance.csv".format(output_dir)
    z_columns = ["z" + str(i) for i in range(coma.z)]
    z_df = pd.DataFrame(None, columns=z_columns)
    perf_df = pd.DataFrame(None, columns=["mse"])

    subsets = ["train", "validation", "test"]
    loader_list = [train_loader, val_loader] if is_this_a_test else [train_loader, val_loader, test_loader]
    all_ids = []
    all_subsets = []

    logging.info("Calculating latent representation (z) and reconstruction performance measures.")

    with torch.no_grad():
        for i, loader in enumerate(loader_list):
            subset = subsets[i]
            all_subsets += [subset] * len(loader.dataset)
            for batch, ids in loader:
                #TODO: We are duplicating code here. The best would be to change encoder according to is_variational
                batch = batch.to(device)
                if coma.is_variational:
                    mu, log_var = coma.encoder(x=batch)
                    z = mu
                else:
                    z = coma.encoder(x=batch)

                z_df = z_df.append(pd.DataFrame(data=z.cpu().detach().numpy(), columns=z_columns))
                all_ids.extend([str(int(x)) for x in ids.numpy()])

                x = coma.decoder(z)

                mse = ((x - batch) ** 2).mean(axis=1).mean(axis=1).cpu().detach().numpy()

                # batch = batch[torch.randperm(len(batch)), :, :]
                #TODO: it's important to report MSE using shuffled data, for reference.
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

    logging.info("Execution finished")

    # This a temporary (i.e. permanent) workaround to indicate that the execution finished
    from subprocess import call
    import shlex
    call(shlex.split("touch %s/.finished" % output_dir))


def train(coma, dataloader, rec_loss_fun, optimizer, device):
    coma.train()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    total_snp_loss = 0
    for i, (data, ids, dosage) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.size(0)
        optimizer.zero_grad()
        out = coma(data)
        recon_loss = rec_loss_fun(out, data.reshape(-1, coma.filters[0]))
        total_recon_loss += batch_size * recon_loss.item()
        loss = recon_loss
        snp_weight = 0.1
        snp_loss = -snp_weight * (dosage - 0.9846).dot(coma.mu[:,5])
        total_snp_loss += batch_size * snp_loss.item()
        loss += snp_loss
        if coma.is_variational:
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kld_loss = -0.5 * torch.mean(torch.mean(1 + coma.log_var - coma.mu ** 2 - coma.log_var.exp(), dim=1), dim=0)
            total_kld_loss += batch_size * kld_loss.item()
            loss += coma.kld_weight * kld_loss            
        total_loss += batch_size * loss.item()
        loss.backward()
        optimizer.step()

    return total_recon_loss / len(dataloader.dataset), \
           total_kld_loss / len(dataloader.dataset), \
           total_snp_loss / len(dataloader.dataset), \
           total_loss / len(dataloader.dataset)


def evaluate(coma, dataloader, rec_loss_fun, device):
    #TODO: condense train and evaluate into a single function with "training" and "evaluate" modes
    coma.eval()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    for i, (data, ids, dosages) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.size(0)
        with torch.no_grad():
            if coma.is_variational:
                mu, log_var = coma.encoder(x=data)
                z = mu # coma.sampling(mu, log_var)
                # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
                kld_loss = -0.5 * torch.mean(torch.mean(1 + coma.log_var - coma.mu ** 2 - coma.log_var.exp(), dim=1), dim=0)
                loss = coma.kld_weight * kld_loss
                total_kld_loss += batch_size * kld_loss.item()
            else:
                z = coma.encoder(x=data)
                loss = 0
            out = coma.decoder(z)
            recon_loss = rec_loss_fun(out, data.reshape(-1, coma.filters[0]))
            total_recon_loss += batch_size * recon_loss.item()
            loss += recon_loss
            total_loss += batch_size * loss.item()

    return total_recon_loss / len(dataloader.dataset), \
           total_kld_loss / len(dataloader.dataset), \
           total_loss / len(dataloader.dataset)


if __name__ == '__main__':

    import argparse
    
    def overwrite_config_items(config, args):
      for attr, value in args.__dict__.items():
        if attr in config.keys() and value is not None:
          config[attr] = value
       
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')

    parser.add_argument('-c', '--conf', help='path of config file', default="config_files/default.config")
    parser.add_argument('-od', '--output_dir', default=None, help='path where to store output')
    parser.add_argument('-id', '--data_dir', default=None, help='path where to fetch input data from')
    parser.add_argument('--preprocessed_data', default=None, type=str, help='Location of cached input data.')
    parser.add_argument('--partition', default=None, type=str, help='Cardiac chamber.')
    parser.add_argument('--procrustes_scaling', default=False, action="store_true",
                        help="Whether to perform scaling transformation after Procrustes alignment (to make mean distance to origin equal to 1).")
    parser.add_argument('--checkpoint_file', default=None, type=str, help='Checkpoint file from which to resume previous training (relative to repository root directory).')
    parser.add_argument('--phase', default=None, help="cardiac phase (1-50|ED|ES)")
    parser.add_argument('--z', default=None, type=int, help='Number of latent variables.')
    parser.add_argument('--optimizer', default=None, type=str, help='optimizer (adam or sgd).')
    parser.add_argument('--epoch', default=None, type=int, help='Maximum number of epochs.')
    parser.add_argument('--nTraining', default=None, type=int, help='Number of training samples.')
    parser.add_argument('--nVal', default=None, type=int, help='Number of validation samples.')
    parser.add_argument('--kld_weight', type=float, default=None, help='Weight of Kullback-Leibler divergence.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate.')
    parser.add_argument('--seed', default=None, help="Seed for PyTorch's Random Number Generator.")
    parser.add_argument('--stop_if_not_learning', default=None, action="store_true",
                        help='Stop training if losses do not change.')
    parser.add_argument('--save_all_models', default=False, action="store_true",
                        help='Save all models instead of just the best one until the current epoch.')
    parser.add_argument('--test', default=False, action="store_true",
                        help='Set this flag if you just want to test whether the code executes properly. ')
    parser.add_argument('--dry-run', dest="dry_run", default=False, action="store_true",
                        help='Dry run: just prints out the parameters of the execution but performs no training.')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        logger.error('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    ################################################################################
    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error('Config not found' + args.conf)
    config = read_config(args.conf)

    if args.data_dir:
        config['data_dir'] = args.data_dir

    if args.output_dir:
        config['output_dir'] = args.output_dir

    if args.test:
        # some small values so that the execution ends quickly
        config['comments'] = "this is a test"
        config['nTraining'] = 500
        config['nVal'] = 80
        config['epoch'] = 20
        config['output_dir'] = "output/test_{TIMESTAMP}"
    config['test'] = args.test
       
    overwrite_config_items(config, args)

    if args.dry_run:
      pprint(config)
      exit()

    main(config)
