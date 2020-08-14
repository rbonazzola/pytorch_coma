# Fork of CoMA for processing of CMR-derived cardiac meshes 

This is a reimplementation of [COMA](https://github.com/anuragranj/coma) to be used for cardiac meshes derived from cardiovascular magnetic resonance (CMR).
Please follow the licensing rights of the authors if you use the code.

## Requirements
This code has been tested on PyTorch v1.3.

Requirements can be met by creating a Conda environment and running the following

```
conda env create -f requirements.yml
```

This will create an environment called `coma` (this name can be changes by editing the `name` field in the above `yaml` file).
Additionally, the following packages, which do not seem to be available in any Conda channel, can be installed with `pip` by executing the following commands:

```
conda environment coma
pip install -f additional_requirements.txt
```

Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh). Note that the python3 version of mesh package library is needed for this.

## Train

The training of the autoencoder is performed by executing the following command:

```
python main.py --config <YAML_FILE>
```

## Data Preparation
### Converting from VTK files to Numpy binary files
The scripts assume that the files containing the 3D mesh information are in binary Numpy format (npy). There is a script in this folder, called `processData.py`, which serves this purpose. To produce these files, run

```
python processData.py \
  --data <SOURCE_DATA> \
  --save_data <OUTPUT_FILE> \
  --partition <PARTITION_NAME> \
  --N_subj <N_SUBJ>
```

where parameters are:
- `data` (string, required): name of a folder containing the VTK files. By default, it is assumed to contain folders, each named as the subject's ID, whose content are the VTK files. So far, only one file per subfolder is supported.
- `save_data` (string, required): output file path. 
- `partition` (string, required): one of LV, LV_endo, LV_epi, RV, LA or RA. 
- `N_subj` (integer, not required): number of subjects considered. By default, it will assume that all the individuals in the `data` folder are to be extracted.

In the case of biventricular or 4-chamber cardiac meshes, typically you would like to extract a single chamber (ventricle or atrium). This is achieved by setting `PARTITION_NAME` as LV[_endo|_epi], RV, LA or RA.

The VTK files derived from CISTIB's segmentation pipeline contain a label for each vertex, indicating which chamber it belongs to (in case of LV, endocardium and epicardium are distinguished); these label are integers between 1 and 5. For more details check out the [VTKHelpers](www.github.com/rodbonazzola/VTKHelpers.git) repository.

The **output file** will be a binary Numpy file containing a tensor of order 3. First axis will represent subjects, second axis means vertex across the mesh, and the third axis represent the spatial coordinate (x, y and z). Also, an additional file containing the subjects' IDs in the same order in which they appear in the tensor.
Note that this script does not produce centered, standardized or Procrustes-aligned data.


### Define the run parameters

By running the following command
` python config_parser.py`
a reference configuration file is created.

As stated above, the run parameters are specified through a `yaml` file. Additional parameters can be provided as command line arguments, in which case the corresponding values in the YAML files would be overwritten, if they already exist there.

The python class `Config` is defined, which allows to nest `yaml` files in a tree-like structure: sections of a yaml file can be provided another `yaml` file path as content. When reading the main (root) `yaml` file by calling the constructor of the class `Config`, the content of the different subfiles are appended to the corresponding sections.
This decision was made in order to facilitate reusability of subsets of run parameters configurations specified in pre-made `yaml` files.


1. **Generate a list of configuration files**: You can use the notebook `notebooks/01_prepare_yaml_files.ipynb` to define parameters for a set of runs. For the user's convenience, a reference file containing default parameters (`config.yaml`) is loaded and a grid of parameters can be defined, where each combination of parameters produces a different `yaml` file.

2. The previous notebook will generate a text file with a list of `yaml` file paths, one per line. It can be used then by the `run_training.sh` bash file, to launch a batch of training executions. If running on an HPC or in the cloud, it's convenient to use e.g. `nohup` or `tmux`, so that the executions are not killed when the session is finished. 


### Tracking the experiments 
Complete this section.
    
## Evaluation
To evaluate on test data you just need to set eval flag to true in default.cfg and provide the path of checkpoint file.


## GWAS
This assumes that the [GWASPipeline](www.github.com/rodbonazzola/GWAS_pipeline.git) has been either downloaded into this repository's root folder, or a symbolic link was created that points to the actual location of the repository in your file system. In Unix-like systems, it could be done by running

```
ln -s <GWAS_PIPELINE_REPO_PATH> GWAS_pipeline
```

## Data preparation
 


#### Note that the pre_transform is provided which normalize the data before storing it. 


## Work to do
-
-
-
