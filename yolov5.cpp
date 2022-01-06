
#include "yolov5/YoloV5Detect.h"
#include <glob.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

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


void run_video()
{
    std::string video_name = "/home/ubuntu/mydata/dataset/video/video2.mp4";
    std::string engine_name = "/home/ubuntu/detect/myinference-new/config/yolov5s.engine";

    YoloV5Detect Detect(engine_name);


    cv::VideoCapture cap(video_name);
    if(!cap.isOpened())
    {
        std::cout<<"error opening video"<<std::endl;
    }
    cv::Mat frame;
    while (1)
    {
        cap>>frame;
        if(frame.empty())
            break;
        std::vector<std::vector<Yolo::Detection>> batch_res = Detect.SingleDetect(frame);

        auto& res = batch_res[0];
        //std::cout << res.size() << std::endl;
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect1(frame, res[j].bbox);
            cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(frame, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }



        cv::imshow("1", frame);
        cv::waitKey(10);

    }

    cap.release();
    Detect.release();


}
void run_imgs()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer!"));
    int color[21][3] =
            {
                    {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                    {0, 0, 255}, {160, 32, 240}, {25, 25, 112},
                    {255, 128, 0}, {189, 252, 201}, {153, 51, 250},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87},
                    {221, 160, 221}, {94, 38, 18}, {0, 199, 140},
                    {0, 0, 255}, {160, 32, 240}, {25, 25, 112},
                    {30, 144, 255}, {0, 199, 140}, {135, 38, 87}
            };

//    std::string imgs_fold = "/home/ubuntu/benewake/benewake_data/img/";
    std::string imgs_fold = "/home/ubuntu/benewake/benewake_data/img_part/";
    std::string engine_name = "/home/ubuntu/detect/myinference-new/config/yolov5s.engine";
    YoloV5Detect Detect(engine_name);

    //fusion with image
    cv::Mat cam_intrinsic, cam_distcoeff, lidar2cam_R, lidar2cam_t;
    cv::Size img_size;
    cv::FileStorage fs_reader("/home/ubuntu/detect/myinference-new/config/bw_fusion.yaml", cv::FileStorage::READ);


    fs_reader["CameraMat"] >> cam_intrinsic;
    fs_reader["DistCoeff"] >> cam_distcoeff;
    fs_reader["ImageSize"] >> img_size;
    fs_reader["lidar2cam_R"]>>lidar2cam_R;
    fs_reader["lidar2cam_t"]>>lidar2cam_t;

    fs_reader.release();

    std::vector<std::string> imgs = glob(imgs_fold+"*.jpg");

    for(auto &e:imgs)
    {
        std::cout<<"img path: "<<e<<std::endl;
        cv::Mat im = cv::imread(e);
        std::vector<std::vector<Yolo::Detection>> batch_res = Detect.SingleDetect(im);
        auto& res = batch_res[0];

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
        std::cout<<"cam_2d size is: "<<res.size()<<std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cv::Mat im_out = im.clone();
        for(int i=0; i<cam_2d.size();i++)
        {
            pcl::PointXYZRGB point;
            point.x = cloud->points[i].x;
            point.y = cloud->points[i].y;
            point.z = cloud->points[i].z;
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect1(im, res[j].bbox);
                cv::rectangle(im_out, r, cv::Scalar(160, 32, 240), 2);
                if(point_rect(cam_2d[i], r))
                {
                    cv::circle(im, cam_2d[i], 1, cv::Scalar(color[j][0], color[j][1], color[j][2]));
                    point.r = color[j][0];
                    point.g = color[j][1];
                    point.b = color[j][2];
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
//        cv::imwrite("/home/ubuntu/CLionProjects/centerpointTrt/onnx_model/out2yolo/"+frame+".jpg", im_out);
        cv::imwrite("/home/ubuntu/benewake/benewake_data/yolov6/"+frame+".jpg", im);
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        viewer->addPointCloud(color_cloud);
        viewer->spin();
//        viewer->spinOnce(50);
    }




}

int main()
{
    run_imgs();
}