# To install PyTorch-geometric and its dependencies, run the following commands.
# CUDA environment variable should match your CUDA version. If you are not using GPUs, but CPUs, set its value to "cpu".

# run these commands on the command line, replace by the corresponding version of each library.
# export CUDA=cu101
# export TORCH=1.4.0

wget https://download.pytorch.org/whl/${CUDA}/torch-${TORCH}-cp36-cp36m-linux_x86_64.whl
pip install torch-${TORCH}-cp36-cp36m-linux_x86_64.whl
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
pip install vtk

# git clone https://github.com/MPI-IS/mesh.git
cd ~/src
git clone git@github.com:MPI-IS/mesh.git
cd -
# Follow instructions in repository to install ^^this^^ library.
