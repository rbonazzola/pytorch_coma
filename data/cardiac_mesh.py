import glob
import os
import re
import time
import numpy as np
import random
from scipy.spatial import procrustes

import sys; sys.path.append("..")
from utils.logger import logger

# I rename the class VTKObject as Mesh so I don't have to change the function calls in the original CoMA code
from VTK.VTKMesh import VTKObject as Mesh
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

    def __init__(self, point_clouds, ids, edges=None,
                 nVal=None, nTraining=None,
                 procrustes_aligned = False,
                 procrustes_scaling=False, procrustes_type="generalized", training_mode=False, is_normalized=False):

        #TODO: add a mesh_partition argument in case the mesh has to be partitioned e.g. for cardiac chamber.

        # self.__ids_file = ids_file
        self.is_normalized = is_normalized

        self.procrustes_aligned = False # procrustes_aligned. Flag to indicate if the mesh set is already aligned
        self.procrustes_scaling = procrustes_scaling
        self.procrustes_type = procrustes_type
        self.edges = edges

        self.training_mode = training_mode # Training or testing

        if isinstance(point_clouds, str) and os.path.isfile(point_clouds):
            self.point_clouds = np.load(point_clouds, allow_pickle=True)
        else:
            self.point_clouds = point_clouds

        if isinstance(ids, str) and os.path.isfile(ids):
            self.ids = [x.strip() for x in open(ids)]
        else:
            self.ids = ids

        self.N, self.n_vertex, self.num_features = self.point_clouds.shape

        if not self.procrustes_aligned:
            self.preprocess_meshes()

        if not self.is_normalized:
            self.normalize()

        # TODO: the partitioning should not be performed in here
        if self.training_mode:
            self.nVal = nVal
            self.nTraining = nTraining
            self.point_clouds_train, self.point_clouds_val, self.point_clouds_test = None, None, None
            self.partition_dataset()


    def preprocess_meshes(self):
        #TODO: Merge these functions into one
        if self.procrustes_scaling:
            self.generalized_procrustes_scaling()
        else:
            self.generalized_procrustes_no_scaling()


    def generalized_procrustes_no_scaling(self):

        logger.info("Performing Procrustes analysis without scaling")
        from scipy.linalg import orthogonal_procrustes

        self.reference_mesh = self.point_clouds[0]
        old_disparity, disparity = 0, 1

        # Center the meshes
        for i in range(len(self.point_clouds)):
            self.point_clouds[i] -= np.mean(self.point_clouds[i], 0)

        it_count = 0
        while abs(old_disparity - disparity) / disparity > 1e-4:
            old_disparity = disparity
            disparity = 0
            for i in range(len(self.point_clouds)):
                # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html
                R, s = orthogonal_procrustes(self.point_clouds[i], self.reference_mesh)

                # Rotate
                self.point_clouds[i] = np.dot(self.point_clouds[i], R) # * s

                # Mean point-wise MSE
                _disparity = np.mean(np.sqrt(np.sum(
                    np.square(self.point_clouds[i] - self.reference_mesh), axis=1)
                ))

                disparity += _disparity

            disparity /= self.point_clouds.shape[0]
            self.reference_mesh = self.point_clouds.mean(axis=0)
            print(disparity)
            it_count += 1

        self.procrustes_aligned = True
        logger.info("Generalized Procrustes analysis performed after %s iterations" % it_count)


    def generalized_procrustes_scaling(self):
        logger.info("Performing Procrustes analysis with scaling")

        old_disparity, disparity = 0, 1  # random values
        self.reference_mesh = self.point_clouds[0]

        it_count = 0
        while abs(old_disparity - disparity) / disparity > 1e-2 and disparity :
            old_disparity = disparity
            disparity = 0
            for i in range(len(self.point_clouds)):
                # Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
                mtx1, mtx2, _disparity = procrustes(
                    self.reference_mesh,
                    self.point_clouds[i]
                )
                disparity += _disparity
                self.point_clouds[i] = np.array(mtx2) # if self.procrustes_scaling else np.array(mtx1)
            disparity /= self.point_clouds.shape[0]
            self.reference_mesh = self.point_clouds.mean(axis=0)
            it_count += 1

        self.procrustes_aligned = True
        logger.info("Generalized Procrustes analysis with scaling performed after %s iterations" % it_count)


    def normalize(self):
        # Mean and std. are computed based on all the samples (not only the training ones). I think this makes sense.
        # Create self.is_normalized argument and set to True to track normalization status.
        # from IPython import embed; embed()
        self.mean, self.std = np.mean(self.point_clouds, axis=0), np.std(self.point_clouds, axis=0)
        self.point_clouds = (self.point_clouds - self.mean) / self.std
        self.is_normalized = True
        logger.info('Vertices normalized')

    #TODO: This should be moved to the dataloader
    def partition_dataset(self):

        ''' Partition full dataset into training, testing and validation subsets '''

        #TODO: Add an option to sample randomly (shuffle)
        self.train_ids = self.ids[:self.nTraining]
        self.val_ids = self.ids[self.nTraining: (self.nTraining + self.nVal)]
        self.test_ids = self.ids[(self.nTraining + self.nVal):]

        #TODO: make this more memory efficient (I am duplicating data here)
        self.point_clouds_train = self.point_clouds[:self.nTraining]
        self.point_clouds_val = self.point_clouds[self.nTraining:(self.nTraining+self.nVal)]
        self.point_clouds_test = self.point_clouds[(self.nTraining+self.nVal):]

        # self.n_vertex = self.point_clouds_train.shape[1]

    def sample(self, BATCH_SIZE=64):
        datasamples = np.zeros((BATCH_SIZE, self.point_clouds_train.shape[1]*self.point_clouds_train.shape[2]))
        for i in range(BATCH_SIZE):
            _randid = random.randint(0, self.N-1)
            datasamples[i] = ((deepcopy(self.point_clouds_train[_randid]) - self.mean) / self.std).reshape(-1)

        return datasamples


    def show(self, ids):
        '''ids: list of ids to play '''
        if max(ids)>=self.N:
            raise ValueError('id: out of bounds')

        mesh = Mesh(v=self.point_clouds_train[ids[0]], f=self.reference_mesh.f)
        time.sleep(0.5)    # pause 0.5 seconds
        viewer = mesh.show()
        for i in range(len(ids)-1):
            viewer.dynamic_meshes = [Mesh(v=self.point_clouds_train[ids[i+1]], f=self.reference_mesh.f)]
            time.sleep(0.5)    # pause 0.5 seconds
        return 0
