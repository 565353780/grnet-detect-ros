pip3 install argparse easydict h5py matplotlib numpy opencv-python pyexr scipy \
  transforms3d tqdm ninja pygments open3d

pip3 install tensorboardX==1.2

# pip3 install -U tensorflow-gpu==2.5.0 keras-nightly==2.5.0.dev2021032900 keras==2.4.3
pip3 install -U torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

cd grnet_detect/src/GRNetDetector/extensions/chamfer_dist
python setup.py install --user

cd ../cubic_feature_sampling
python setup.py install --user

cd ../gridding
python setup.py install --user

cd ../gridding_loss
python setup.py install --user

