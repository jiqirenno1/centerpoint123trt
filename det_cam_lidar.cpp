//
// Created by ubuntu on 2021/12/2.
//
#include "centerpoint.h"
#include "yolov5/YoloV5Detect.h"
#include <glob.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include "tracking/Mot3D.h"

std::vector<std::string> glob(const std::string pattern)
{
    std::vector<std::string> filenames;
    using namespace std;
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0){
        globfree(&glob_result);
        return filenames;
    }
    for(auto idx =0; idx <glob_result.gl_pathc; idx++){
        filenames.push_back(string(glob_result.gl_pathv[idx]));

    }
    globfree(&glob_result);
    return filenames;
}

cv::Rect get_rect1(cv::Mat& img, float bbox[4]) {
    int l, r, t, b; //left right top bottom
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

bool point_rect(cv::Point2d & pp, cv::Rect& rect)
{
    double l = rect.x;
    double t = rect.y;
    double r = l + rect.width;
    double b = t + rect.height;
    if(pp.x>=l&&pp.x<=r&&pp.y>=t&&pp.y<=b)
        return true;
    return false;
}
bool point_box(pcl::PointXYZ& pt, Box& box)
{
    double l = box.y - box.w/2; //left
    double r = box.y + box.w/2; //right
    double bk = box.x - box.l/2;//back
    double f = box.x + box.l/2;//front
    double bt = box.z - box.h/2;//bottom
    double t = box.z + box.h/2;//top

    double xx = pt.x;
    double yy = pt.y;
    double zz = pt.z;
    if(xx>=bk&&xx<=f&&yy>=l&&yy<=r&&zz>=bt&&zz<=t)
        return true;
    return false;
}
int getIouNum(std::vector<int>& aa, std::vector<int>& bb)
{
    int num = 0;
    for(auto&a:aa)
    {
        for(auto&b:bb)
        {
            if(a==b)
                num++;
        }
    }
    return num;
}

void singleCloudDetect(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::visualization::PCLVisualizer::Ptr &viewer, std::vector<std::vector<int>>& cam_cloud_id, std::shared_ptr<CenterPoint>& cp_ptr)
{
//    viewer->addPointCloud(cloud);
    std::vector<Box> predResult = cp_ptr->singleInference(cloud);
    int num = predResult.size();

    if(num==0)
        return;
    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> cloud_id(num, std::vector<int>());
    for(int i=0;i<cloud->size();i++)
    {
        pcl::PointXYZ pp = cloud->points[i];
        for(int j=0;j<num;j++)
        {
            Box bb = predResult[j];
            if(point_box(pp, bb))
            {
                cloud_id[j].push_back(i);
                break;
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
    std::cout<<"***********cloud cls id  is: "<<time1<<std::endl;

    startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> match_id(cam_cloud_id.size(), std::pair<int, int>(-1, -1));
    for(int i=0;i<cam_cloud_id.size();i++)
    {
        std::vector<int> cam_cluster = cam_cloud_id[i];
        int iouNums = 0;
        for(int j=0;j<cloud_id.size();j++)
        {
            int tempNums = getIouNum(cam_cluster, cloud_id[j]);
            if(tempNums>iouNums)
            {
                match_id[i] = std::pair<int, int>(i, j);
                iouNums = tempNums;
            }
        }

    }
    endTime = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
    std::cout<<"***********cloud match  id  is: "<<time2<<std::endl;

    int color[21][3] =
            {
                    {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                    {0, 0, 255}, {0, 255, 255}, {255, 0, 255},
                    {255, 128, 0}, {189, 252, 201}, {153, 51, 250},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87},
                    {221, 160, 221}, {94, 38, 18}, {0, 199, 140},
                    {0, 0, 255}, {160, 32, 240}, {25, 25, 112},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87}
            };
    int name =1000;
    for(auto&e:match_id)
    {
        if(e.second==-1)
            continue;
        name++;
        Box bb = predResult[e.second];
        viewer->addCube(bb.x - bb.l/2, bb.x + bb.l/2, bb.y - bb.w/2, bb.y + bb.w/2, bb.z - bb.h/2, bb.z + bb.h/2, color[e.first][0], color[e.first][1], color[e.first][2], to_string(name));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, to_string(name));
//        pcl::PointXYZ pText;
//        pText.x = bb.x - bb.l/2;
//        pText.y = bb.y - bb.w/2;
//        pText.z = bb.z - bb.h/2;
//        viewer->addText3D(to_string(e.second), pText, 1.5, 1, 0, 0, "info"+to_string(name));

    }



//    for(int i=0; i<num; i++)
//    {
//        Box bb = predResult[i];
//        viewer->addCube(bb.x - bb.l/2, bb.x + bb.l/2, bb.y - bb.w/2, bb.y + bb.w/2, bb.z - bb.h/2, bb.z + bb.h/2, 0, 1, 0, to_string(i));
//        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, to_string(i));
//    }
}

void run_imgs()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer!"));
    int color[21][3] =
            {
                    {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                    {0, 0, 255}, {0, 255, 255}, {255, 0, 255},
                    {255, 128, 0}, {189, 252, 201}, {153, 51, 250},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87},
                    {221, 160, 221}, {94, 38, 18}, {0, 199, 140},
                    {0, 0, 255}, {160, 32, 240}, {25, 25, 112},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87}
            };

//    std::string imgs_fold = "/home/ubuntu/benewake/benewake_data/img/";
    std::string imgs_fold = "/home/ubuntu/benewake/benewake_data/img_part/";

    //init yolov5
    std::string engine_name = "../onnx_model/yolov5/yolov5s.engine";
    std::shared_ptr<YoloV5Detect> yolo_ptr = std::make_shared<YoloV5Detect>(engine_name);

    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
    cp_ptr->loadFromEngine("../onnx_model/cp.engine");

    //fusion param
    cv::Mat cam_intrinsic, cam_distcoeff, lidar2cam_R, lidar2cam_t;
    cv::Size img_size;
    cv::FileStorage fs_reader("../onnx_model/bw_fusion.yaml", cv::FileStorage::READ);
    fs_reader["CameraMat"] >> cam_intrinsic;
    fs_reader["DistCoeff"] >> cam_distcoeff;
    fs_reader["ImageSize"] >> img_size;
    fs_reader["lidar2cam_R"]>>lidar2cam_R;
    fs_reader["lidar2cam_t"]>>lidar2cam_t;
    fs_reader.release();
    //

    std::vector<std::string> imgs = glob(imgs_fold+"*.jpg");

    for(auto &e:imgs)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        std::cout<<"img path: "<<e<<std::endl;
        cv::Mat im = cv::imread(e);
        std::vector<std::vector<Yolo::Detection>> batch_res = yolo_ptr->SingleDetect(im);
        auto& res = batch_res[0];
        auto endTime = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        std::cout<<"***********img inference time is: "<<time1<<std::endl;

        auto npos = e.find_last_of('/');
        auto frame = e.substr(npos + 1, 4);
        std::string pcd_path = "/home/ubuntu/benewake/benewake_data/pcd/"+frame+".pcd";
        std::cout<<"pcd path: "<<pcd_path<<std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(pcd_path, *cloud)) {
            std::cout << "error when loading: " << pcd_path << std::endl;
        }
        std::cout<<"cloud size: "<<cloud->size()<<std::endl;


        std::vector<cv::Point3d> lidar_3d;
        std::vector<cv::Point2d> cam_2d;
        for(auto&e:cloud->points)
        {
            lidar_3d.emplace_back(e.x, e.y, e.z);
        }
        cv::projectPoints(lidar_3d, lidar2cam_R, lidar2cam_t, cam_intrinsic, cam_distcoeff, cam_2d);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cv::Mat im_out = im.clone();

        startTime = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> cam_cloud_id(res.size(), std::vector<int>());
        for(int i=0; i<cam_2d.size();i++)
        {
            pcl::PointXYZRGB point;
            point.x = cloud->points[i].x;
            point.y = cloud->points[i].y;
            point.z = cloud->points[i].z;
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect1(im, res[j].bbox);
                //cv::rectangle(im_out, r, cv::Scalar(160, 32, 240), 2);
                if(point_rect(cam_2d[i], r))
                {
                    //cv::circle(im, cam_2d[i], 1, cv::Scalar(color[j][0], color[j][1], color[j][2]));
                    point.r = color[j][0];
                    point.g = color[j][1];
                    point.b = color[j][2];
                    cam_cloud_id[j].push_back(i);
                    break;
                }
                else
                {
                    point.r = 255;
                    point.g = 255;
                    point.b = 255;
                }
            }
            color_cloud->points.push_back(point);

        }

        endTime = std::chrono::high_resolution_clock::now();
        double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        std::cout<<"***********caculate time  is: "<<time2<<std::endl;
//        cv::imwrite("/home/ubuntu/CLionProjects/centerpointTrt/onnx_model/out2yolo/"+frame+".jpg", im_out);
//        cv::imwrite("/home/ubuntu/benewake/benewake_data/yolov6/"+frame+".jpg", im);
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        viewer->addPointCloud(color_cloud, "fusion");
        startTime = std::chrono::high_resolution_clock::now();
        singleCloudDetect(cloud, viewer, cam_cloud_id, cp_ptr);
        endTime = std::chrono::high_resolution_clock::now();
        double time3 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        std::cout<<"***********cloud inference time is: "<<time3<<std::endl;
//        viewer->spin();
        viewer->spinOnce();
    }




}

int main()
{
    run_imgs();
}

void det2d()
{
    //init yolov5
    std::string engine_name = "../onnx_model/yolov5/yolov5s.engine";
    YoloV5Detect Detect(engine_name);
    while(1)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        cv::Mat im = cv::imread("/home/ubuntu/benewake/benewake_data/img_part/img/1000.jpg");
        std::vector<std::vector<Yolo::Detection>> batch_res = Detect.SingleDetect(im);
        auto& res = batch_res[0];
        auto endTime = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        std::cout<<"***********img inference time is: "<<time1<<std::endl;
    }

}
void det3d()
{
    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
    cp_ptr->loadFromEngine("../onnx_model/cp.engine");
    std::string path = "/home/ubuntu/benewake/benewake_data/img_part/pcd/1000.pcd";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(path, *cloud)) {
        std::cout << "error when loading: " <<path<< std::endl;
    }
    while(1)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Box> predResult = cp_ptr->singleInference(cloud);
        auto endTime = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        std::cout<<"***********cloud inference time is: "<<time1<<std::endl;

    }


}

int main2()
{

//    std::thread det2(det2d);
    std::thread det3(det3d);
//    det2.join();
    det3.join();


}

void do_process1(cv::Mat cloud)
{

}

void do_process2(cv::Mat &cloud)
{

}

void do_process3(std::shared_ptr<cv::Mat> &cloud)
{

}

int maineee()
{
    auto startTime0 = std::chrono::high_resolution_clock::now();
    std::string path = "/home/ubuntu/benewake/benewake_data/img_part/pcd/1000.pcd";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(path, *cloud)) {
        std::cout << "error when loading: " <<path<< std::endl;
    }
    auto endTime0 = std::chrono::high_resolution_clock::now();
    cv::Mat im = cv::imread("/home/ubuntu/benewake/benewake_data/img_part/img/1000.jpg");
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime0 - startTime0).count()/1000000.0;
    std::cout<<"***********process time1 is: "<<time<<std::endl;

    Mot3D track3D;
    std::shared_ptr<Mot3D> trk3d = std::make_shared<Mot3D>();
    std::shared_ptr<cv::Mat> imm = std::make_shared<cv::Mat>();
    auto startTime = std::chrono::high_resolution_clock::now();
    int i=10000;
    while(i)
    {
        do_process1(im);
        i--;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
    std::cout<<"***********process time1 is: "<<time1<<std::endl;

    auto startTime2 = std::chrono::high_resolution_clock::now();
    int i2=10000;
    while(i2)
    {
        do_process2(im);
        i2--;
    }
    auto endTime2 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime2 - startTime2).count()/1000000.0;
    std::cout<<"***********process time2 is: "<<time2<<std::endl;


    auto startTime3 = std::chrono::high_resolution_clock::now();
    int i3=10000;
    while(i3)
    {
        do_process3(imm);
        i3--;
    }
    auto endTime3 = std::chrono::high_resolution_clock::now();
    double time3 = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime3 - startTime3).count()/1000000.0;
    std::cout<<"***********process time3 is: "<<time3<<std::endl;

}



