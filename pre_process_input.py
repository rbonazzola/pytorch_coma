import pickle
from cardiac_mesh import CardiacMesh
from config_parser import read_config
from helpers import get_template_mesh

config = read_config("config_files/default.cfg")

dataset = CardiacMesh(
  meshes_file=config['data_dir'],
  ids_file=config['ids_file'],
  reference_mesh=get_template_mesh(config),
  procrustes_scaling=config["procrustes_scaling"],
  procrustes_type=config["procrustes_type"],
  mode="validation"
)

pickle.dump(dataset, open("data/meshes/numpy_files/LV_all_subjects/LV_GPA_meshes.pkl", "wb"))