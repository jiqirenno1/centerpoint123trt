//
// Created by ubuntu on 8/12/21.
//

#include "YoloV5Detect.h"
#include "common.hpp"

// stuff we know about the network and the input/output blobs
const int YoloV5Detect::INPUT_H = Yolo::INPUT_H;
const int YoloV5Detect::INPUT_W = Yolo::INPUT_W;
const int YoloV5Detect::CLASS_NUM = Yolo::CLASS_NUM;
const int YoloV5Detect::OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;


static Logger gLogger;

YoloV5Detect::YoloV5Detect(std::string engine_name) {
    //1. read engine deserialize the .engine
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
//    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

}

std::vector<std::vector<Yolo::Detection>> YoloV5Detect::SingleDetect(cv::Mat img) {
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    auto start = std::chrono::system_clock::now();
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox
    int i = 0;
    int b=0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    auto end = std::chrono::system_clock::now();
    std::cout <<"preprocess time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Run inference
//        auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    end = std::chrono::system_clock::now();
    std::cout  <<"preprocess+inference time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(BATCH_SIZE);

    nms(batch_res[0], &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);

    return batch_res;
}

void
YoloV5Detect::doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output,
                          int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void YoloV5Detect::release() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

}

YoloV5Detect::~YoloV5Detect() {

}

