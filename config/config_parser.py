import os
import configparser
import yaml

def get_repo_rootdir():
    import shlex
    from subprocess import check_output
    repo_rootdir = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
    return repo_rootdir


def is_yaml_file(x):
    if isinstance(x, str):
        return x.endswith("yaml") or x.endswith("yml")
    return False


def unfold_config(token):
    '''
    Parameters: a recursive structure composed of a path to a yaml file or a dictionary composed of such structures.
    Returns: A dictionary with all the yaml files replaces by their content.
    '''
    repo_rootdir = get_repo_rootdir()
    yaml_dir = os.path.join(repo_rootdir, "config_files")
    if is_yaml_file(token):
        #TODO: COMMENT AND DOCUMENT THIS!!!
        try:
            token = yaml.safe_load(open(token))
        except FileNotFoundError:
            kk = open(os.path.join(yaml_dir, token))
            token = yaml.safe_load(kk)
    if isinstance(token, dict):
        for k, v in token.items():
            token[k] = unfold_config(v)
    return token


def read_config(fname):

    import json

    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return

    config = json.load(open(fname, "rt"))
    
    # Coerce the items into the correct data types

    config['seed'] = int(config['seed'])
    config['procrustes_scaling'] = bool(config['procrustes_scaling'])

    config['n_layers'] = int(config['n_layers'])
    config['z'] = int(config['z'])
    
    # config['downsampling_factors'] =  [int(x) for x in config['downsampling_factors'].split(',')]
    # config['num_conv_filters'] = [int(x) for x in config['num_conv_filters'].split(',')]
    # config['polygon_order'] = [int(x) for x in config['polygon_order'].split(',')]
    
    config['workers_thread'] = int(config['workers_thread'])

    config['kld_weight'] = float(config['kld_weight'])

    config['weight_loss'] = config.get('Model Parameters', 'weight_loss') # TODO: add this option into the scripts

    config['nVal'] = int(config['nVal'])
    config['nTraining'] = int(config['nTraining'])
    config['batch_size'] = int(config['batch_size'])
    config['learning_rate'] = float(config['learning_rate'])
    config['learning_rate_decay'] = float(config['learning_rate_decay'])
    config['weight_decay'] = float(config['weight_decay'])
    config['epoch'] = int(config['epoch'])

    config['stop_if_not_learning'] = bool(config.get('stop_if_not_learning', True))
    config['save_all_models'] = bool(config.get('save_all_models', False))

    return(config)

# def read_config(fname):

#     if not os.path.exists(fname):
#         print('Config not found %s' % fname)
#         return

#     config = configparser.RawConfigParser()
#     config.read(fname)

#     self = {}
#     try:
#       self['seed'] = config.getint('Seed', 'seed')
#     except ValueError:
#       self['seed'] = None


#     self['procrustes_scaling'] = config.getboolean('Pre-processing Parameters', 'procrustes_scaling')

#     self['checkpoint_file'] = config.get('Model Parameters', 'checkpoint_file')

#     self['n_layers'] = config.getint('Model Parameters', 'n_layers')
#     self['z'] = config.getint('Model Parameters', 'z')
#     self['downsampling_factors'] =  [int(x) for x in config.get('Model Parameters', 'downsampling_factors').split(',')]
#     self['num_conv_filters'] = [int(x) for x in config.get('Model Parameters', 'num_conv_filters').split(',')]
#     self['polygon_order'] = [int(x) for x in config.get('Model Parameters', 'polygon_order').split(',')]
#     self['optimizer'] = config.get('Model Parameters', 'optimizer')

#     self['workers_thread'] = config.getint('Model Parameters', 'workers_thread')

#     self['activation_function'] = config.get('Model Parameters', 'activation_function') # TODO: add this option into the scripts
#     self['reconstruction_loss'] = config.get('Model Parameters', 'reconstruction_loss') # TODO: add this option into the scripts
#     self['kld_weight'] = config.getfloat('Model Parameters', 'kld_weight') 

#     self['weight_loss'] = config.get('Model Parameters', 'weight_loss')                 # TODO: add this option into the scripts

#     self['nVal'] = config.getint('Learning Parameters', 'nVal') 
#     self['nTraining'] = config.getint('Learning Parameters', 'nTraining') 
#     self['batch_size'] = config.getint('Learning Parameters', 'batch_size')
#     self['learning_rate'] = config.getfloat('Learning Parameters', 'learning_rate')
#     self['learning_rate_decay'] = config.getfloat('Learning Parameters', 'learning_rate_decay')
#     self['weight_decay'] = config.getfloat('Learning Parameters', 'weight_decay')
#     self['epoch'] = config.getint('Learning Parameters', 'epoch')

#     self['comments'] = config.get('Additional', 'comments')                             # TODO: add this option into the scripts
#     self['group_label'] = config.get('Additional', 'group_label')                       # TODO: add this option into the scripts
#     self['label'] = config.get('Additional', 'label')                                   # TODO: add this option into the scripts
#     self['stop_if_not_learning'] = config.get('Additional', 'stop_if_not_learning')  # TODO: add this option into the scripts
#     self['save_all_models'] = config.get('Additional', 'save_all_models')               # TODO: add this option into the scripts

#     return self


def save_config(config, filename):

    print('Writing default config file - %s' % filename)
    with open(filename, 'w') as configfile:
        config.write(configfile)
        configfile.close()


def save_default_config(replace=True):
    config_fname = "../config_files/default.cfg"
    if os.path.exists(config_fname) and replace or not os.path.exists(config_fname):
        config = configparser.RawConfigParser()
        set_default_parameters(config)
        save_config(config, config_fname)
    return config_fname


def read_default_config():
    config_fname = save_default_config()
    return read_config(config_fname)

if __name__ == '__main__':

    pkg_path, _ = os.path.split(os.path.realpath(__file__))

    if not os.path.exists("config_files"):
        os.makedirs("config_files")

    config_fname = os.path.join(pkg_path, '../config_files/default.cfg')
    config = configparser.RawConfigParser()
    set_default_parameters(config)

    save_config(config, config_fname)
