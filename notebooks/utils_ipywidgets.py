import ipywidgets as widgets
from IPython.display import display
import os

def select_experiment(finished=True, selected="2020-09-11_02-13-41"):
  output_dir = "../pytorch_coma/output/"
  if finished:
    experiments = [x for x in sorted(os.listdir(output_dir)) if os.path.exists(os.path.join(output_dir, x, ".finished"))]
  else:
    experiments = [x for x in os.listdir(output_dir) if os.path.exists(os.path.join(output_dir, x))]
  w = widgets.Dropdown(
    value='2020-09-11_02-13-41',
    options=experiments,
    description='Experiment:',
    disabled=False,
  )
  return w

def pvplot_mesh(meshes, notebook=True, **kargs):
  import pyvista as pv
  kargs = {"point_size": 5, "render_points_as_spheres": True}
  plotter = pv.Plotter(notebook=notebook)
  if isinstance(meshes, list):
    for mesh in meshes:
      plotter.add_mesh(mesh, **kargs)
  else:
    plotter.add_mesh(meshes, **kargs)
  plotter.show(interactive=True)
  plotter.enable()

def ids_widget(meshes):
  wcb = widgets.Combobox(
    placeholder="Choose a subject",
    options=sorted(meshes.ids),
    value="1000336"
  )
  return wcb
