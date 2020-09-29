import argparse
import pickle
from cardiac_mesh import *
import sys; sys.path.append("..")
from utils.logger import logger

class VTKDataset(object):
    """This function generates Numpy binary data files from a set of VTK files"""

    def __init__(self, folders,
                 filename_pattern,
                 partition_ids=None,
                 subj_ids=None, N_subj=None):

        '''
          - subj_ids: list of subject IDs to draw samples from (if None, it's all the subjects)
          - N_subj: number of subjects to sample (if None, it's all the subjects from subj_ids)
        '''

        # TODO: comment which are the assumptions regarding the file names
        self.folders = folders if isinstance(folders, list) else [folders]
        self.filename_pattern = os.path.join(folders, filename_pattern)

        # This line is to replace e.g. "*/*/*" with "(.*)/(.*)/(.*)"
        self.regex = re.compile(self.filename_pattern.replace("*", "(.*)"))

        self.partition_ids = partition_ids
        self.N_subj = N_subj
        self.subj_ids = subj_ids

        self.gather_paths()
        self.ids, self.point_clouds, self.edges = self.gather_data()

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
            ids = list(path_dict.keys())  # [extract_id_from_path(x) x for x in datapaths]
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

        ids = []
        vertices = []

        mesh = Mesh(filename=self.datapaths[0], load_connectivity=True)
        mesh = Mesh.extractSubpart(mesh, self.partition_ids)
        edges = mesh.edges
        shape = mesh.points.shape

        # tqdm: for progress bar
        for i, p in enumerate(tqdm(self.datapaths, unit="subjects")):
            mesh_filename = p
            mesh = Mesh(filename=mesh_filename, load_connectivity=False)  # Load mesh
            mesh = Mesh.extractSubpart(mesh, self.partition_ids)
            if mesh.points.shape == shape:
                vertices.append(mesh.points)
                ids.append(self.subj_ids[i])
            else:
                logger.error("Individual {}".format(self.subj_ids[i]))

        return ids, np.array(vertices), edges


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


def CardiacDataset(data_path, partition, subj_ids=None, N_subj=None):

    # if not os.path.exists(args.save_path):
    # os.makedirs(args.save_path)

    logger.info("Pre-processing meshes...")

    vtk_dataset = VTKDataset(
        folders=data_path,
        filename_pattern="*/output.001.vtk",
        # subj_ids=args.subj_ids, N_subj=args.N_subj,
        partition_ids=CardiacMesh.SUBPARTS_IDS[args.partition]
    )

    cardiacDataset = CardiacMesh(
        point_clouds=vtk_dataset.point_clouds,
        edges=vtk_dataset.edges,
        ids=vtk_dataset.ids,
        procrustes_scaling=args.scaled
    )
    return cardiacDataset


def main(args):

    if args.save_path is None:
        o_file = "transforms/cached/{dataset}__{partition}__{phase}__{scaled}.pkl".format(
            dataset=args.dataset,
            partition=args.partition,
            phase="ED" if args.phase == 1 else str(args.phase),
            scaled="scaled" if args.scaled else "non_scaled"
        )
    elif args.save_path.endswith("pkl"):
        o_file = args.save_path
    else:
        raise ArgumentError

    if os.path.exists(o_file) and not args.replace_previous:
        logger.error("Pickle file already exists. Aborting process.")
        exit()

    logger.info("Caching pickle file {}...".format(o_file))
    cardiacDataset = CardiacDataset(args.data_folder, args.partition, args.scaled)
    pickle.dump(cardiacDataset, open(o_file, "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocessing data for Convolutional Mesh Autoencoders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_folder', type=str, default="datasets/2ch_segmentation",
        help='path to the data directory'
    )

    parser.add_argument(
        '--dataset', dest='dataset', type=str, default="2ch_segmentation",
        help='Name of the dataset'
    )

    parser.add_argument(
        '--partition', dest='partition', type=str, default="LV",
        help='partition of the mesh, i.e. chamber(s) of the heart'
    )

    parser.add_argument(
        '--scaled', dest='scaled', action="store_true", default=False,
        help='A boolean indicating whether the mesh are scaled or not when performing Procrustes alignment.'
    )

    parser.add_argument(
        '--phase', dest='phase', type=int, default=1,
        help='An integer between 1 and 50.'
    )

    parser.add_argument(
        '--save_path', dest='save_path', type=str, default=None,
        help='path where processed data will be saved'
    )

    parser.add_argument(
        '--replace_previous', action="store_true", default=False,
        help='whether to replace a previously generated file or not'
    )

    args = parser.parse_args()

    main(args)