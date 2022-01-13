# GRNet-Detect-Ros

## Download model
```bash
https://gateway.infinitescript.com/?fileName=GRNet-ShapeNet.pth
```

and save it to
```bash
~/.ros/
```

## Install
```bash
pip3 install argparse easydict h5py matplotlib numpy opencv-python pyexr scipy tensorboardX==1.2 transforms3d tqdm ninja pygments open3d==0.10.0.0
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

mkdir -p grnet_ws/src
cd grnet_ws/src
git clone https://github.com/565353780/grnet-detect-ros.git
cd grnet-detect-ros/grnet_detect/src/GRNetDetector/extensions/chamfer_dist
python setup.py install --user

cd ../cubic_feature_sampling
python setup.py install --user

cd ../gridding
python setup.py install --user

cd ../gridding_loss
python setup.py install --user

cd ../../../../..
```

## Build
```bash
cd grnet_ws
catkin init
catkin config --cmake-args -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=Yes
catkin build grnet-detect-ros
```

## Service
name
```bash
grnet_detect/detect
```
service req&res
```bash
grnet_detect/srv/PC2ToPC2.srv
```

## Enjoy it~

