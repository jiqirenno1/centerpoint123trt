//
// Created by ubuntu on 8/12/21.
//

#ifndef YOLOV5_YOLOV5DETECT_H
#define YOLOV5_YOLOV5DETECT_H

#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "NvInfer.h"
#include "yololayer.h"
#include "utils.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

using namespace nvinfer1;

class YoloV5Detect {
public:
    YoloV5Detect() = default;
    explicit YoloV5Detect(std::string engine_name);
    ~YoloV5Detect();
public:
    std::vector<std::vector<Yolo::Detection>> SingleDetect(cv::Mat img);

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    void release();

public:
    static const int INPUT_H;
    static const int INPUT_W;
    static const int CLASS_NUM;
    static const int OUTPUT_SIZE;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";

private:
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    void* buffers[2];

};


#endif //YOLOV5_YOLOV5DETECT_H
