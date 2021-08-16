# TSDF Fusion (CPU Version)

This is a CPU version of the TSDF fusion developed by Andy Zeng. The original CUDA accelerated version can be found [here](https://github.com/andyzeng/tsdf-fusion).

## Installation
```shell
mkdir build && cd build
cmake ..
make
```
Required open-cv version: 4.2.0
## Usage
1. Prepare the camera intrinsic file. An example of camera intrinsic file is in ps-instrinsic.txt
2. Prepare the data. Save the depth image and the homogenous transformation of the camera frame in the world frame when taking the image. The name of the depth image and its corresponding pose txt should be the same. Example data are saved in data/. Example output is saved in model/.
3. Run the executable
```shell
./build/tsdf_fusion_cpu cam_K_file data_dir num_of_frame v_size v_x_dim v_y_dim v_z_dim v_x_origin v_y_origin v_z_origin
```
Example runing command:
```shell
./build/tsdf-fusion-cpu ps-intrinsic.txt data/ 24 0.002 200 200 100 -0.4 -0.5 0.01
```



