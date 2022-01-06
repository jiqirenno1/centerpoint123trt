//
// Created by ubuntu on 6/22/21.
//

#include "tracking/TrackLet.h"

int TrackLet::count_ = 0;
TrackLet::TrackLet(Eigen::Vector3d &det, double time) {
    t_ = time;
    count_+=1;
    id_ = count_;

    Eigen::VectorXd X(6); // x,y,x, vx,vy,vz
    X<< det[0], det[1], det[2], 0, 0, 0;
    KF_.Initialization(X);

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd block_F = Eigen::MatrixXd::Identity(3,3);
    F.block<3,3>(0,3) = block_F;
    KF_.SetF(F);

    Eigen::MatrixXd u(6,1);
    u << 0, 0, 0, 0, 0, 0;
    KF_.SetU(u);
    KF_.SetB(Eigen::MatrixXd::Identity(6,6));

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd block_P = Eigen::MatrixXd::Identity(3,3) * 1000 ;
    P.block<3,3>(3,3) = block_P;
    KF_.SetP(P);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6,6);
    Eigen::MatrixXd block_Q = Eigen::MatrixXd::Identity(3,3) * 0.01 ;
    Q.block<3,3>(3,3) = block_Q;
    KF_.SetQ(Q);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 6);
    H.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
    KF_.SetH(H);
    KF_.SetR(Eigen::MatrixXd::Identity(3, 3));

}

TrackLet::~TrackLet() {

}

void TrackLet::predict(double time) {
    //cancle
//    double diffT = time - t_;
//    t_ = time;
//    std::cout<<" ******deta time: "<<diffT<<std::endl;
//    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6,6);
//    Eigen::MatrixXd block_F = Eigen::MatrixXd::Identity(3,3)*diffT;
//    F.block<3,3>(0,3) = block_F;
//    KF_.SetF(F);

    KF_.Prediction();
    time_since_update_+=1;

}

void TrackLet::update(Eigen::Vector3d &det) {
    detZ_ = det.z() - posZ_;
    posZ_ = detZ_;

    KF_.KFUpdate(det);
    hits_+=1;
    time_since_update_=0;

}

Eigen::Vector3d TrackLet::getState() {

    return KF_.GetX().head(3);
}

int TrackLet::getID() {
    return id_;
}

float TrackLet::getSpeed() {
//    return int(KF_.GetX()[3]);
    std::cout<<"speed : "<<std::sqrt(std::pow(KF_.GetX()[3], 2) + std::pow(KF_.GetX()[4], 2) + std::pow(KF_.GetX()[5], 2))<<std::endl;
    std::cout<<"time_since_update_: "<<time_since_update_<<std::endl;
    std::cout<<"hits_ : "<<hits_<<std::endl;
    return std::sqrt(std::pow(KF_.GetX()[3], 2) + std::pow(KF_.GetX()[4], 2) + std::pow(KF_.GetX()[5], 2));
}

float TrackLet::smoothSpeed() {
    if(meanSpeed==0)
        meanSpeed = getSpeed();
    else
        meanSpeed = meanSpeed*0.8 + getSpeed()*0.2;
    return meanSpeed;
}

