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
mkdir -p grnet_ws/src
cd grnet_ws/src
git clone https://github.com/565353780/grnet-detect-ros.git
cd grnet-detect-ros
./setup.sh
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

