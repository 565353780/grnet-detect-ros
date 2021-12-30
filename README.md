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
git clone https://github.com/565353780/grnet-detect-ros.git
cd grnet-detect-ros/src/GRNetDetector/
pip install -r requirements.txt
cd extensions/chamfer_dist
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

