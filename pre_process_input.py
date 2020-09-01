import pickle
from cardiac_mesh import CardiacMesh
from config_parser import read_config
from helpers import get_template_mesh

config = read_default_config()

dataset = CardiacMesh(
  meshes_file=config['data_dir'],
  ids_file=config['ids_file'],
  reference_mesh=get_template_mesh(config),
  procrustes_scaling=config["procrustes_scaling"],
  procrustes_type=config["procrustes_type"],
  mode="validation"
)

pickle.dump(dataset, open(config['preprocessed_data'], "wb"))