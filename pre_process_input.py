import pickle
from cardiac_mesh import CardiacMesh
from helpers import get_template_mesh

import configparser
config = configparser.RawConfigParser()
config = set_default_parameters(config)

dataset = CardiacMesh(
  meshes_file=config['data_dir'],
  ids_file=config['ids_file'],
  reference_mesh=get_template_mesh(config),
  procrustes_scaling=config["procrustes_scaling"],
  procrustes_type=config["procrustes_type"],
  mode="validation"
)

pickle.dump(dataset, open(config['preprocessed_data'], "wb"))
