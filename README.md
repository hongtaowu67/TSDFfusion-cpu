# TSDF Fusion (CPU Version)

This is a CPU version of the TSDF fusion developed by Andy Zeng. The original CUDA accelerated version can be found [here](https://github.com/andyzeng/tsdf-fusion).

## Installation
```shell
mkdir build && cd build
cmake ..
make
```

## Usage
1. Prepare the camera intrinsic file. An example of camera intrinsic file is in ps-instrinsic.txt
2. Run the executable
```shell
./build/tsdf_fusion_cpu cam_K_file data_dir num_of_frame v_size v_x_dim v_y_dim v_z_dim v_x_origin v_y_origin v_z_origin
```
