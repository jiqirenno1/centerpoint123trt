//
// Created by ubuntu on 2021/11/25.
//

#include "centerpoint.h"
#include "tracking/Mot3D.h"

void singleDetect(string path, pcl::visualization::PCLVisualizer::Ptr &viewer, Mot3D& track3D)
{
    std::vector<Eigen::Vector3d> dets;
    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
    cp_ptr->loadFromEngine("../onnx_model/cp.engine");

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
        Eigen::Vector3d deti;
        deti << bb.x, bb.y, bb.z;
        dets.push_back(deti);
    }

    vector<pair<int, float>> idspeed = track3D.update(dets, 1);

    for(int i=0;i<idspeed.size();i++)
    {
        if(idspeed[i].first!=-1)
        {
            Box bb = predResult[i];
            viewer->addCube(bb.x - bb.l/2, bb.x + bb.l/2, bb.y - bb.w/2, bb.y + bb.w/2, bb.z - bb.h/2, bb.z + bb.h/2, 0, 1, 0, to_string(i));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, to_string(i));
            pcl::PointXYZ pText;
            pText.x = bb.x - bb.l/2;
            pText.y = bb.y - bb.w/2;
            pText.z = bb.z - bb.h/2;

//            viewer->addText3D("Id: "+to_string(idspeed[i].first), pText, 2, 1, 0, 0, "Id"+to_string(bb.z));
            viewer->addText3D("Id: "+to_string(idspeed[i].first)+"| Speed: "+to_string(std::abs(int(idspeed[i].second*3.6*6))), pText, 1.5, 1, 0, 0, "info"+to_string(i));

        }

    }
}

//int main(int argc, char* argv[])
int main()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer!"));
    string dataPath = "/home/ubuntu/benewake/benewake_data/pcd";
    std::cout<<"dataPath: "<<dataPath<<std::endl;
    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath},
                                               boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    auto streamIterator = paths.begin();

    Mot3D track3D;

    while(!viewer->wasStopped())
    {
        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        // Load pcd and run obstacle detection process
        singleDetect(streamIterator->string(), viewer, track3D);
        streamIterator++;
        if (streamIterator == paths.end()) {
            streamIterator = paths.begin();
        }
        viewer->spinOnce(30);
    }
}