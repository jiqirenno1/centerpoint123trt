//
// Created by ubuntu on 2021/11/23.
//

#ifndef CENTERPOINTTRT_CENTERPOINT_H
#define CENTERPOINTTRT_CENTERPOINT_H

#include <iostream>

#include "common.h"
#include "argsParser.h"
#include "parserOnnxConfig.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>

class CenterPoint {
    template<typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
    CenterPoint();
    bool buildFromOnnx(const std::string& onnx_file, const std::string& engine_file, bool save);
    bool loadFromEngine(const std::string& engine_file);
    std::vector<Box> singleInference(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
private:
    samplesCommon::OnnxSampleParams mParams;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser, std::string onnx_file);



};


#endif //CENTERPOINTTRT_CENTERPOINT_H
