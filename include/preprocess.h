#ifndef __CENTERPOINT_PREPROCESS__
#define __CENTERPOINT_PREPROCESS__
#include <iostream>
#include <fstream>
#include "config.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

void preprocess(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float* feature, int* indices, int pointNum);

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum);

template <typename T>
bool saveBinFile(std::string savePath, T* output, size_t shape)
{
    //Save one out node
    std::fstream file(savePath, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cout << "Error opening file." << savePath << std::endl;;
        return false;
    }
    file.write(reinterpret_cast<char *>(output), shape*sizeof(T));
    file.close();
    return true;
}
#endif