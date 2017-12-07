## 1. Installation
Install tensorflow, for instructions please follow https://www.tensorflow.org/install

Recommended tensorflow version: r1.2

Install numpy, matplotlib, Pillow:

**sudo pip install numpy pillow matplotlib**

## 2. Preparing to run the Program
Download [TensorFlow model](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy)

## 3. Run the model
**python predict.py {model_path} {image_path}**

## 4. Export the model and executing it
**python exportModel.py {model_path} {export_dir}**

**python inference.py {model_dir} {image_dir}**