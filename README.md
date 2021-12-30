# GRNet-Detect-Ros

## Install
your catkin_ws should looks like
```bash
catkin_ws/
|----src/
     |----grnet-detect-ros/
|----devel/
|----build/
|----logs/
```

run this to install

```bash
pip3 install argparse easydict h5py matplotlib numpy opencv-python pyexr scipy tensorboardX==1.2 transforms3d tqdm ninja pygments open3d==0.10.0.0
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

git clone https://github.com/565353780/grnet-detect-ros.git

cd grnet-detect-ros/src/GRNetDetector/extensions/chamfer_dist
python setup.py install --user

cd ../cubic_feature_sampling
python setup.py install --user

cd ../gridding
python setup.py install --user

cd ../gridding_loss
python setup.py install --user

cd ../../../../
catkin build
```

## Service
name
```bash
grnet_detect/detect
```
service req&res
```bash
srv/PC2ToPC2.srv
```

## Enjoy it~

