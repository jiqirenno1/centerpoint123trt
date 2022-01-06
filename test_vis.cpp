//
// Created by ubuntu on 2021/11/23.
//
#include "centerpoint.h"

void singleDetect(string path, pcl::visualization::PCLVisualizer::Ptr &viewer, std::shared_ptr<CenterPoint>& cp_ptr)
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(path, *cloud)) {
        std::cout << "error when loading: " <<path<< std::endl;
    }
    viewer->addPointCloud(cloud);
    std::vector<Box> predResult = cp_ptr->singleInference(cloud);
    int num = predResult.size();
    for(int i=0; i<num; i++)
    {
        Box bb = predResult[i];
        std::cout << "bb.l:" <<bb.l<<" bb.w:"<<bb.w<<" bb.h:"<<bb.h<<" bb.x:"<<bb.x<<" bb.y:"<<bb.y<<" bb.z:"<<bb.z<<" bb.theta:"<<bb.theta<<" bb.cls:"<<bb.cls<< std::endl;
        viewer->addCube(bb.x - bb.l/2, bb.x + bb.l/2, bb.y - bb.w/2, bb.y + bb.w/2, bb.z - bb.h/2, bb.z + bb.h/2, 0, 1, 0, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, to_string(i));
    }
}

//int main(int argc, char* argv[])
int main_1()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer!"));
    string dataPath = "/home/ubuntu/lidar_annotation/test";
    std::cout<<"dataPath: "<<dataPath<<std::endl;
    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath},
                                               boost::filesystem::directory_iterator{});

    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
    cp_ptr->loadFromEngine("../onnx_model/cp.engine");

    sort(paths.begin(), paths.end());

    auto streamIterator = paths.begin();


    while(!viewer->wasStopped())
    {
        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        // Load pcd and run obstacle detection process
        singleDetect(streamIterator->string(), viewer, cp_ptr);
        streamIterator++;
        if (streamIterator == paths.end()) {
            streamIterator = paths.begin();
        }
        viewer->spin();
    }
}

int main()
{
    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
    std::string onnx_file = "/home/ubuntu/PycharmProjects/det3/CenterPoint/onnx_model/pointpillars_trt.onnx";
    std::string engine_file = "../onnx_model/cp_multihead.engine";
    cp_ptr->buildFromOnnx(onnx_file, engine_file, true);

}
