// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// -----
// Modifier: Hongtao Wu, Johns Hopkins University, 2020
// Modification: Able to fuse data from two cameras
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "boost/filesystem.hpp"

#include "utils.hpp"

using namespace boost::filesystem;

void Integrate(float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  for (int pt_grid_x=0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {
    for (int pt_grid_y=0; pt_grid_y < voxel_grid_dim_y; ++pt_grid_y){
        for (int pt_grid_z=0; pt_grid_z < voxel_grid_dim_z; ++pt_grid_z){
          // Convert voxel center from grid coordinates to base frame camera coordinates
          float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
          float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
          float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

          // Convert from base frame camera coordinates to current frame camera coordinates
          float tmp_pt[3] = {0};
          tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
          tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
          tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
          float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
          float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
          float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

          if (pt_cam_z <= 0)
            continue;

          int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
          int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
          if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
            continue;

          float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

          if (depth_val <= 0 || depth_val > 6)
            continue;

          float diff = depth_val - pt_cam_z;

          if (diff <= -trunc_margin)
            continue;

          // Integrate
          int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
          float dist = fmin(1.0f, diff / trunc_margin);
          float weight_old = voxel_grid_weight[volume_idx];
          float weight_new = weight_old + 1.0f;
          voxel_grid_weight[volume_idx] = weight_new;
          voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
      }
    }
  }
}

// Loads a binary file with depth data and generates a TSDF voxel volume
// Fuse the volumn captured by two camera. One is mounted on the end-effector of a robot arm, the other is fixed.
// ./tsdf_fusion_dual_camera cam_K_file_1 cam_K_file_2 data_path
// data is formated as 000000_depth.png, 000000_pose.txt, 000000_depth_fix.png, 000000_pose_fix.txt
int main(int argc, char * argv[]) {

  // Location of camera intrinsic file
  std::string cam_K_file_1;
  std::string cam_K_file_2;

  cam_K_file_1 = argv[1];
  cam_K_file_2 = argv[2];

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path;

  data_path = argv[3];

  int base_frame_idx = 0;
  int first_frame_idx = 0;

  float cam_K_1[3 * 3];
  float cam_K_2[3 * 3];
  float base2world[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -0.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -0.5f;
  float voxel_grid_origin_z = -0.1f;
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 100;
  int voxel_grid_dim_y = 100;
  int voxel_grid_dim_z = 50;

  // Manual parameters
  if (argc > 4) {
    int counter = 4;
    std::cout << "Parsing input parameter...\n";

    voxel_size = atof(argv[counter]);
    counter++;

    voxel_grid_dim_x = std::atoi(argv[counter]);
    counter++;

    voxel_grid_dim_y = std::atoi(argv[counter]);
    counter++;

    voxel_grid_dim_z = std::atoi(argv[counter]);
    counter++;

    voxel_grid_origin_x = std::atof(argv[counter]);
    counter++;

    voxel_grid_origin_y = std::atof(argv[counter]);
    counter++;

    voxel_grid_origin_z = std::atof(argv[counter]);
    counter++;

    std::cout << "Finish parsing input parameter!\n";
  }

  std::cout << "cam_K_file_1: " << cam_K_file_1 << std::endl;
  std::cout << "cam_K_file_2: " << cam_K_file_2 << std::endl;

  // Read camera intrinsics
  std::vector<float> cam_K_vec_1 = LoadMatrixFromFile(cam_K_file_1, 3, 3);
  std::vector<float> cam_K_vec_2 = LoadMatrixFromFile(cam_K_file_2, 3, 3);
  std::copy(cam_K_vec_1.begin(), cam_K_vec_1.end(), cam_K_1);
  std::copy(cam_K_vec_2.begin(), cam_K_vec_2.end(), cam_K_2);

  std::cout << "Camera Intrinsic 1: " << cam_K_vec_1[0] << ", " << cam_K_vec_1[1] << ", " << cam_K_vec_1[2] << std::endl;
  std::cout << "Camera Intrinsic 2: " << cam_K_vec_2[0] << ", " << cam_K_vec_2[1] << ", " << cam_K_vec_2[2] << std::endl;

  // Initialize voxel grid
  float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Loop throught the data directory
  std::vector<std::string> file_path_list_1; // depth files from the first cam
  std::vector<std::string> file_path_list_2; // depth_files from the second cam

  std::string substring1 = "depth.png";
  std::string substring2 = "depth_fix.png";

  path p(data_path);
  directory_iterator end_itr;

  for(directory_iterator itr(p); itr != end_itr; ++itr)
  {
    if (is_regular_file(itr->path()))
    {
      std::string current_file = itr->path().string();
      
      if (current_file.find(substring1) != std::string::npos)
        file_path_list_1.push_back(current_file);
      if (current_file.find(substring2) != std::string::npos)
        file_path_list_2.push_back(current_file);
    }
  }

  int frame_num_1 = file_path_list_1.size();
  int frame_num_2 = file_path_list_2.size();

  std::cout << "Num of frames from moving cam: " << frame_num_1 << std::endl;
  std::cout << "Num of frames from fixed cam: " << frame_num_2 << std::endl;

  // Loop through the images captured by the fixed camera
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + frame_num_2; ++frame_idx) {
    
    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

    // Read current frame depth
    std::string depth_im_file = data_path + "/" + curr_frame_prefix.str() + "_depth_fix.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);

    // Read base frame camera pose
    std::string cam2world_file = data_path + "/" + curr_frame_prefix.str() + "_pose_fix.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate(cam_K_2, cam2world, depth_im,
              im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
              voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
              voxel_grid_TSDF, voxel_grid_weight);
  }

  // Loop through the images capture by the camera on the end-effector
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + frame_num_1; ++frame_idx) {
    
    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

    // Read current frame depth
    std::string depth_im_file = data_path + "/" + curr_frame_prefix.str() + "_depth.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);

    // Read base frame camera pose
    std::string cam2world_file = data_path + "/" + curr_frame_prefix.str() + "_pose.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate(cam_K_1, cam2world, depth_im,
              im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
              voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
              voxel_grid_TSDF, voxel_grid_weight);
  }

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  std::string ply_file_path = data_path + "/tsdf.ply";
  SaveVoxelGrid2SurfacePointCloud(ply_file_path, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

  // Save TSDF voxel grid and its parameters to disk as binary file (float array)
  std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
  std::string voxel_grid_saveto_path = data_path + "/tsdf.bin";
  std::ofstream outFile(voxel_grid_saveto_path.c_str(), std::ios::binary | std::ios::out);
  float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
  float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
  float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
  outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
  outFile.write((char*)&voxel_size, sizeof(float));
  outFile.write((char*)&trunc_margin, sizeof(float));
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i) {
    outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
  }
  outFile.close();

  return 0;
}


