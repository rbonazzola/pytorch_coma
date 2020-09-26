import glob
import os
import re
import time
import numpy as np
import random
from scipy.spatial import procrustes
from Logger import logger

# I rename the class VTKObject as Mesh so I don't have to change the function calls in the original CoMA code
from VTK.VTKMesh import VTKObject as Mesh  #, MeshViewer, MeshViewers
from copy import deepcopy
from tqdm import tqdm # for the progress bar

__author__ = "Rodrigo Bonazzola"

#TODO: Create a class called Mesh and make CardiacMesh inherit from it, adding the specific features of the cardiac mesh into the later.

class CardiacMesh(object):

    SUBPARTS_IDS = {
        "all": None,
        "LV": [1, 2],
        "LV_endo": [1],
        "LV_epi": [2],
        "LA": [3],
        "RV": [4],
        "RA": [5],
    }

    def __init__(self, meshes_file,
                 reference_mesh, ids_file=None,
                 nVal=None, nTraining=None,
                 subpart=None,
                 procrustes_aligned = False,
                 procrustes_scaling=False, procrustes_type="generalized", mode='training', is_normalized=False):

        #TODO: add a mesh_partition argument in case the mesh has to be partitioned e.g. for cardiac chamber.

        self.__meshes_file = meshes_file
        self.__ids_file = ids_file
        self.is_normalized = is_normalized

        self.procrustes_aligned = procrustes_aligned # Flag to indicate if the mesh set is already aligned
        self.procrustes_scaling = procrustes_scaling
        self.procrustes_type = procrustes_type

        # self.N = None # This is just a placeholder now. The actual value will be filled in later.
        # self.n_vertex = None # idem
        # self.num_features = None
        self.__mode = mode # Training or testing

        self.reference_mesh = reference_mesh
        
        self.load_meshes()

        if not self.procrustes_aligned:
            self.preprocess_meshes()

        if not self.is_normalized:
            self.normalize()

        # TODO: the partitioning should not be performed in here
        if self.__mode == 'training':
            self.nVal = nVal
            self.nTraining = nTraining
            self.vertices_train, self.vertices_val, self.vertices_test = None, None, None
            self.partition_dataset()


    def preprocess_meshes(self):

        # Merge these functions into one
        if self.procrustes_scaling:
            self.generalized_procrustes_scaling()
        else:
            self.generalized_procrustes_no_scaling()

    def generalized_procrustes_no_scaling(self):

        logger.info("Performing Procrustes analysis without scaling")
        from scipy.linalg import orthogonal_procrustes

        reference_point_cloud = self.reference_mesh.points  # reference to align to
        old_disparity, disparity = 0, 1

        # Center the meshes
        for i in range(len(self.vertices)):
            self.vertices[i] -= np.mean(self.vertices[i], 0)

        it_count = 0
        while abs(old_disparity - disparity) / disparity > 1e-4:
            old_disparity = disparity
            disparity = 0
            for i in range(len(self.vertices)):
                # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html
                R, s = orthogonal_procrustes(self.vertices[i], reference_point_cloud)

                # Rotate
                self.vertices[i] = np.dot(self.vertices[i], R) # * s

                # Mean point-wise MSE
                _disparity = np.mean(np.sqrt(np.sum(
                    np.square(self.vertices[i] - reference_point_cloud), axis=1)
                ))

                disparity += _disparity

            disparity /= self.vertices.shape[0]
            reference_point_cloud = self.vertices.mean(axis=0)
            print(disparity)
            it_count += 1

        self.procrustes_aligned = True
        logger.info("Generalized Procrustes analysis performed after %s iterations" % it_count)

    def generalized_procrustes_scaling(self):
        logger.info("Performing Procrustes analysis with scaling")

        old_disparity, disparity = 0, 1  # random values
        reference_point_cloud = self.reference_mesh.points  # reference to align to

        it_count = 0
        while abs(old_disparity - disparity) / disparity > 1e-2 and disparity :
            old_disparity = disparity
            disparity = 0
            for i in range(len(self.vertices)):
                # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
                mtx1, mtx2, _disparity = procrustes(
                    reference_point_cloud,
                    self.vertices[i]
                )
                disparity += _disparity
                self.vertices[i] = np.array(mtx2) # if self.procrustes_scaling else np.array(mtx1)
            disparity /= self.vertices.shape[0]
            reference_point_cloud = self.vertices.mean(axis=0)
            it_count += 1

        self.procrustes_aligned = True
        logger.info("Generalized Procrustes analysis with scaling performed after %s iterations" % it_count)


    def load_meshes(self):

        ''' Load numpy data files containing mesh data and a list of subject IDs. '''

        self.vertices = np.load(self.__meshes_file, allow_pickle=True) # training + validation + testing
        self.ids = [x.strip() for x in open(self.__ids_file)]
        self.N, self.n_vertex, self.num_features = self.vertices.shape


    def partition_dataset(self):

        ''' Partition full dataset into training, testing and validation subsets '''

        # TODO: Add an option to sample randomly (shuffle)
        self.train_ids = self.ids[:self.nTraining]
        self.val_ids = self.ids[self.nTraining: (self.nTraining + self.nVal)]
        self.test_ids = self.ids[(self.nTraining + self.nVal):]

        #TODO: make this more memory efficient (I am duplicating data here)
        self.vertices_train = self.vertices[:self.nTraining]
        self.vertices_val = self.vertices[self.nTraining:(self.nTraining+self.nVal)]
        self.vertices_test = self.vertices[(self.nTraining+self.nVal):]

        # self.n_vertex = self.vertices_train.shape[1]


    def normalize(self):
        # Mean and std. are computed based on all the samples (not only the training ones). I think this makes sense.
        # Create self.is_normalized argument and set to True to track normalization status.
        # from IPython import embed; embed()
        self.mean, self.std = np.mean(self.vertices, axis=0), np.std(self.vertices, axis=0)
        self.vertices = (self.vertices - self.mean) / self.std
        self.is_normalized = True
        logger.info('Vertices normalized')



    def sample(self, BATCH_SIZE=64):
        datasamples = np.zeros((BATCH_SIZE, self.vertices_train.shape[1]*self.vertices_train.shape[2]))
        for i in range(BATCH_SIZE):
            _randid = random.randint(0, self.N-1)
            datasamples[i] = ((deepcopy(self.vertices_train[_randid]) - self.mean) / self.std).reshape(-1)

        return datasamples

    def vec2mesh(self, vec):
        vec = vec.reshape((self.n_vertex, 3))*self.std + self.mean
        return Mesh(v=vec, f=self.reference_mesh.f)


    def show(self, ids):
        '''ids: list of ids to play '''
        if max(ids)>=self.N:
            raise ValueError('id: out of bounds')

        mesh = Mesh(v=self.vertices_train[ids[0]], f=self.reference_mesh.f)
        time.sleep(0.5)    # pause 0.5 seconds
        viewer = mesh.show()
        for i in range(len(ids)-1):
            viewer.dynamic_meshes = [Mesh(v=self.vertices_train[ids[i+1]], f=self.reference_mesh.f)]
            time.sleep(0.5)    # pause 0.5 seconds
        return 0

    def save_meshes(self, filename, meshes):
        for i in range(meshes.shape[0]):
            vertices = meshes[i].reshape((self.n_vertex, 3)) * self.std + self.mean
            mesh = Mesh(v=vertices, f=self.reference_mesh.f)
            # TODO: replace the write function
            mesh.write_ply(filename+'-'+str(i).zfill(3)+'.ply')
        return 0

    def show_mesh(self, viewer, mesh_vecs, figsize):
        for i in range(figsize[0]):
            for j in range(figsize[1]):
                mesh_vec = mesh_vecs[i*(figsize[0]-1) + j]
                mesh_mesh = self.vec2mesh(mesh_vec)
                viewer[i][j].set_dynamic_meshes([mesh_mesh])
        time.sleep(0.1)
        return 0

    def get_normalized_meshes(self, mesh_paths):
        meshes = []
        for mesh_path in mesh_paths:
            mesh = Mesh(filename=mesh_path)
            mesh_v = (mesh.v - self.mean)/self.std
        meshes.append(mesh_v)
        return np.array(meshes)


class NumpyFromVTKs(object):
    
    """This function generates Numpy binary data files from a set of VTK files"""

    def __init__(self, folders, 
                   filename_pattern, 
                   dataset_name, 
                   partition_ids=None, 
                   subj_ids=None, N_subj=None):
        
        '''          
          - subj_ids: list of subject IDs to draw samples from (if None, it's all the subjects)
          - N_subj: number of subjects to sample (if None, it's all the subjects from subj_ids)
        '''
       
        #TODO: comment which are the assumptions regarding the file names
        self.folders = folders if isinstance(folders, list) else [folders]
        self.filename_pattern = os.path.join(folders, filename_pattern)

        # This line is to replace e.g. "*/*/*" with "(.*)/(.*)/(.*)"
        self.regex = re.compile(self.filename_pattern.replace("*","(.*)"))
        
        self.dataset_name = dataset_name # output directory
        
        self.partition_ids = partition_ids
        self.N_subj = N_subj
        self.subj_ids = subj_ids

        self.gather_paths()
        self.vertices = self.gather_data()
       
           
    def extract_id_from_path(self, path):
        id = self.regex.match(path).group(1)
        return id
        
        
    def gather_paths(self):
        
        '''
          Gather all the paths associated to mesh files
        '''

        datapaths = []
        for subdir_name in self.folders:
            datapaths.extend(sorted(glob.glob(self.filename_pattern)))
        
        path_dict = {self.extract_id_from_path(x): x for x in datapaths}
            
        if self.subj_ids is None:
            ids = path_dict.keys() # [extract_id_from_path(x) x for x in datapaths]
        else:
            ids = [x for x in path_dict.keys() if x in subj_ids]
                                        
        if self.N_subj is not None:
            random.shuffle(ids)            
            # TODO: add sanity check for case when N_subj > len(ids)
            ids = ids[0:self.N_subj]        
            
        datapaths = [path_dict[i] for i in ids]
        self.subj_ids = ids
        self.datapaths = datapaths


    def gather_data(self):
        vertices = []

        # tqdm: for progress bar
        for p in tqdm(self.datapaths, unit="subjects"):
            mesh_filename = p
            mesh = Mesh(filename=mesh_filename, load_connectivity=False) # Load mesh
            mesh = Mesh.extractSubpart(mesh, self.partition_ids)
            vertices.append(mesh.points)

        return np.array(vertices)


    def save_vertices(self, output_filename, ids_filename=None):

        if not os.path.exists(os.path.dirname(output_filename)):
            logger.info("Creating directory %s" % os.path.dirname(output_filename))
            os.makedirs(os.path.dirname(output_filename))

        if not output_filename.endswith(".npy"):
            output_filename += ".npy"

        if ids_filename is None:
            ids_filename = output_filename if not output_filename.endswith(".npy") else output_filename[:-4]
            ids_filename += "_subject_ids.txt"

        if not os.path.exists(output_filename) and not os.path.exists(ids_filename):
            np.save(output_filename, self.vertices)
            with open(ids_filename, "w") as ff:
                ff.write("\n".join(self.subj_ids))
        else:
            logger.error("File already exists")
            return 1

        logger.info("Numpy mesh file created in %s" % output_filename)
        logger.info("Subject IDs file created in %s" % ids_filename)

        return 0
    
    
def generateNumpyDataset(data_path, save_path, partition, subj_ids=None, N_subj=None):

    logger.info("Generating numpy binary data files from VTK meshes")

    npy = NumpyFromVTKs(
      folders=data_path,
      filename_pattern="*/output.001.vtk",
      dataset_name=save_path,
      subj_ids=subj_ids,
      N_subj=N_subj, # all
      partition_ids=CardiacMesh.subparts_ids[partition]
    )

    npy.save_vertices(save_path)
    
    return 0
