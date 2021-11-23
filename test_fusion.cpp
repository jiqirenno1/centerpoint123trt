//
// Created by ubuntu on 2021/11/23.
//

#include "centerpoint.h"
#include <glob.h>
#include <opencv2/opencv.hpp>

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

std::vector<cv::Point3d> bbox2points(Box & box)
{
    cv::Mat pos;
    cv::Mat Cl=(cv::Mat_<double>(1,8)<<1,  1,  1,  1, -1, -1, -1, -1);
    cv::Mat Cw=(cv::Mat_<double>(1,8)<<1, -1, -1,  1,  1, -1, -1,  1);
    cv::Mat Ch=(cv::Mat_<double>(1,8)<<1,  1, -1, -1,  1,  1, -1, -1);

    Cl = Cl*box.l + box.x;
    Cw = Cw*box.w + box.y;
    Ch = Ch*box.h + box.z;

    std::vector<cv::Mat> vpos;
    vpos.push_back(Cl);
    vpos.push_back(Cw);
    vpos.push_back(Ch);

    cv::vconcat(vpos, pos);

    std::vector<cv::Point3d> res_3d;
    int cols = pos.cols;
    for(int i=0; i<cols; i++)
    {
        res_3d.emplace_back(pos.at<double>(0, i), pos.at<double>(1, i), pos.at<double>(2, i));
    }
    res_3d.emplace_back(box.x, box.y, box.z);
    return res_3d;
}
int main()
{
    std::shared_ptr<CenterPoint> cp_ptr = std::make_shared<CenterPoint>();
//    cp_ptr->buildFromOnnx("../onnx_model/pointpillars_trt.onnx", "./cp.engine",
//                          true);
    cp_ptr->loadFromEngine("../onnx_model/cp.engine");

    //fusion with image
    cv::Mat cam_intrinsic, cam_distcoeff, lidar2cam_R, lidar2cam_t;
    cv::Size img_size;
    cv::FileStorage fs_reader("../onnx_model/bw_fusion.yaml", cv::FileStorage::READ);


    fs_reader["CameraMat"] >> cam_intrinsic;
    fs_reader["DistCoeff"] >> cam_distcoeff;
    fs_reader["ImageSize"] >> img_size;
    fs_reader["lidar2cam_R"]>>lidar2cam_R;
    fs_reader["lidar2cam_t"]>>lidar2cam_t;

    fs_reader.release();

    std::vector<std::string> filePath = glob("/home/ubuntu/benewake/benewake_data/pcd/*.pcd");

    for(auto idx = 0; idx < filePath.size(); idx++) {
        std::cout << "filePath[idx]: " << filePath[idx] << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(filePath[idx], *cloud)) {
            std::cout << "error when loading: " << filePath[idx] << std::endl;
            return false;
        }

        std::vector<Box> predResult = cp_ptr->singleInference(cloud);

        auto npos = filePath[idx].find_last_of('/');
        auto frame = filePath[idx].substr(npos + 1, 4);
        std::cout<<"frame is : "<<frame<<std::endl;
        cv::Mat img = cv::imread("/home/ubuntu/benewake/benewake_data/img/"+frame+".jpg");
        int i=-1;
        for(auto &box:predResult)
        {
            i++;
//            cv::Mat img1 = img.clone();
            std::vector<cv::Point3d> lidar_3d = bbox2points(box);
            std::vector<cv::Point2d> cam_2d;
            cv::projectPoints(lidar_3d, lidar2cam_R, lidar2cam_t, cam_intrinsic, cam_distcoeff, cam_2d);

            for(int i=0;i<cam_2d.size();i++)
            {
                if(cam_2d[i].x>=0&&cam_2d[i].x<img_size.width&&cam_2d[i].y>=0&&cam_2d[i].y<img_size.height)
                {
                    if(i == cam_2d.size()-1)
                    {
                        cv::circle(img, cam_2d[i], 3, {0, 0, 255});
                    }
                    else{
                        cv::circle(img, cam_2d[i], 1, {0, 255, 0});
                    }

                }
            }

//            cv::imwrite(std::to_string(i)+".jpg", img1);

        }
        cv::imwrite("/home/ubuntu/CLionProjects/centerpointTrt/onnx_model/out/"+frame+".jpg", img);
    }
}