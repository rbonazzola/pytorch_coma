import os
import configparser


def set_default_parameters(config):

    config.add_section('Seed')
    config.set('Seed', 'seed')

    config.add_section('Input Output')
    config.set('Input Output', 'data_dir', 'data/meshes/numpy_files/LV_all_subjects/train.npy')
    config.set('Input Output', 'preprocessed_data', 'data/transforms/cached/2ch_segmentation__LV__ED__non_scaled.pkl')
    config.set('Input Output', 'template_fname', './template/template.vtk')
    config.set('Input Output', 'output_dir', 'output/{TIMESTAMP}')
    config.set('Input Output', 'partition', 'LV')

    # TODO: add these options into the scripts
    config.add_section('Pre-processing Parameters')
    config.set('Pre-processing Parameters', 'procrustes_scaling', 'False')

    config.add_section('Model Parameters')
    config.set('Model Parameters', 'checkpoint_file', '')
    config.set('Model Parameters', 'n_layers', '4')
    config.set('Model Parameters', 'z', '8')
    config.set('Model Parameters', 'downsampling_factors', '4, 4, 3, 2')
    config.set('Model Parameters', 'num_conv_filters', '16, 16, 16, 32, 32')
    config.set('Model Parameters', 'polygon_order', '6, 6, 6, 6, 6')        # TODO: Polygon?? Shouldn't it be "polynomial"
    config.set('Model Parameters', 'workers_thread', 8)
    config.set('Model Parameters', 'optimizer', 'adam')
    config.set('Model Parameters', 'activation_function', 'relu')         # TODO: add this option into the scripts
    config.set('Model Parameters', 'reconstruction_loss', 'mse')           # TODO: add this option into the scripts
    config.set('Model Parameters', 'kld_weight', 0) 
    config.set('Model Parameters', 'weight_loss', 0)                  # TODO: add this option into the scripts

    config.add_section('Learning Parameters')
    config.set('Learning Parameters', 'nTraining', 1000)
    config.set('Learning Parameters', 'nVal', 1000) 
    config.set('Learning Parameters', 'batch_size', 16)
    config.set('Learning Parameters', 'learning_rate', 8e-3)
    config.set('Learning Parameters', 'learning_rate_decay', 0.99)
    config.set('Learning Parameters', 'weight_decay', 5e-4) ### What's this??
    config.set('Learning Parameters', 'epoch', 300)


    config.add_section('Additional')
    config.set('Additional', 'comments', '') # Some kind of reminder of why this experiment was run
    config.set('Additional', 'group_label', '') # A label for the set of runs to which this run belongs
    config.set('Additional', 'label', '')  # A label for this particular run
    config.set('Additional', 'save_all_models', False)
    config.set('Additional', 'stop_if_not_learning', True)


def read_config(fname):

    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return

    config = configparser.RawConfigParser()
    config.read(fname)

    self = {}
    try:
      self['seed'] = config.getint('Seed', 'seed')
    except ValueError:
      self['seed'] = None

    self['data_dir'] = config.get('Input Output', 'data_dir')
    self['preprocessed_data'] = config.get('Input Output', 'preprocessed_data')
    self['template_fname'] = config.get('Input Output', 'template_fname')
    self['output_dir'] = config.get('Input Output', 'output_dir')
    self['partition'] = config.get('Input Output', 'partition')

    self['procrustes_scaling'] = config.getboolean('Pre-processing Parameters', 'procrustes_scaling')

    self['checkpoint_file'] = config.get('Model Parameters', 'checkpoint_file')
    self['n_layers'] = config.getint('Model Parameters', 'n_layers')
    self['z'] = config.getint('Model Parameters', 'z')
    self['downsampling_factors'] =  [int(x) for x in config.get('Model Parameters', 'downsampling_factors').split(',')]
    self['num_conv_filters'] = [int(x) for x in config.get('Model Parameters', 'num_conv_filters').split(',')]
    self['polygon_order'] = [int(x) for x in config.get('Model Parameters', 'polygon_order').split(',')]
    self['workers_thread'] = config.getint('Model Parameters', 'workers_thread')
    self['optimizer'] = config.get('Model Parameters', 'optimizer')

    self['activation_function'] = config.get('Model Parameters', 'activation_function') # TODO: add this option into the scripts
    self['reconstruction_loss'] = config.get('Model Parameters', 'reconstruction_loss') # TODO: add this option into the scripts
    self['kld_weight'] = config.getfloat('Model Parameters', 'kld_weight') 

    self['weight_loss'] = config.get('Model Parameters', 'weight_loss')                 # TODO: add this option into the scripts

    self['nVal'] = config.getint('Learning Parameters', 'nVal') 
    self['nTraining'] = config.getint('Learning Parameters', 'nTraining') 
    self['batch_size'] = config.getint('Learning Parameters', 'batch_size')
    self['learning_rate'] = config.getfloat('Learning Parameters', 'learning_rate')
    self['learning_rate_decay'] = config.getfloat('Learning Parameters', 'learning_rate_decay')
    self['weight_decay'] = config.getfloat('Learning Parameters', 'weight_decay')
    self['epoch'] = config.getint('Learning Parameters', 'epoch')

    self['comments'] = config.get('Additional', 'comments')                             # TODO: add this option into the scripts
    self['group_label'] = config.get('Additional', 'group_label')                       # TODO: add this option into the scripts
    self['label'] = config.get('Additional', 'label')                                   # TODO: add this option into the scripts
    self['stop_if_not_learning'] = config.get('Additional', 'stop_if_not_learning')  # TODO: add this option into the scripts
    self['save_all_models'] = config.get('Additional', 'save_all_models')               # TODO: add this option into the scripts

    return self


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


class Config:

    def __init__(self, config_fname):

        raise NotImplementedError

        #TODO: finish this and replace the current approach
        config = configparser.RawConfigParser()
        config.read(config_fname)

        self.visualize = config.getboolean('Input Output', 'visualize')
        self.data_dir = config.get('Input Output', 'data_dir')
        self.checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
        self.template_fname = config.get('Input Output', 'template_fname')
        self.visual_output_dir = config.get('Input Output', 'visual_output_dir')

        self.procrustes_type = config.get('Pre-processing Parameters', 'procrustes_type')
        self.procrustes_scaling = config.getboolean('Pre-processing Parameters', 'procrustes_scaling')

        self.eval = config.getboolean('Model Parameters', 'eval')
        self.z = config.getint('Model Parameters', 'z')
        self.downsampling_factors = [int(x) for x in
                                                config.get('Model Parameters', 'downsampling_factors').split(',')]
        self.num_conv_filters = [int(x) for x in
                                            config.get('Model Parameters', 'num_conv_filters').split(',')]
        self.polygon_order = [int(x) for x in config.get('Model Parameters', 'polygon_order').split(',')]
        self.workers_thread = config.getint('Model Parameters', 'workers_thread')
        self.optimizer = config.get('Model Parameters', 'optimizer')

        self.activation_function = config.get('Model Parameters',
                                                         'activation_function')  # TODO: add this option into the scripts
        self.reconstruction_loss = config.get('Model Parameters',
                                                         'reconstruction_loss')  # TODO: add this option into the scripts
        self.variational_loss = config.get('Model Parameters',
                                                      'variational_loss')  # TODO: add this option into the scripts
        self.weight_loss = config.get('Model Parameters',
                                                 'weight_loss')  # TODO: add this option into the scripts

        self.nVal = config.getint('Learning Parameters', 'nVal')  # TODO: add this option into the scripts
        self.nTraining = config.getfloat('Learning Parameters',
                                                    'nTraining')  # TODO: add this option into the scripts
        self.batch_size = config.getint('Learning Parameters', 'batch_size')
        self.learning_rate = config.getfloat('Learning Parameters', 'learning_rate')
        self.learning_rate_decay = config.getfloat('Learning Parameters', 'learning_rate_decay')
        self.weight_decay = config.getfloat('Learning Parameters', 'weight_decay')
        self.epoch = config.getint('Learning Parameters', 'epoch')

        self.comments = config.get('Additional', 'comments')  # TODO: add this option into the scripts
        self.group_label = config.get('Additional', 'group_label')  # TODO: add this option into the scripts
        self.label = config.get('Additional', 'label')  # TODO: add this option into the scripts
