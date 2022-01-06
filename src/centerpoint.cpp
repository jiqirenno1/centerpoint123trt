//
// Created by ubuntu on 2021/11/23.
//

#include "centerpoint.h"

CenterPoint::CenterPoint() {
    //init mParams
    mParams.inputTensorNames.emplace_back("input.1");
    mParams.inputTensorNames.emplace_back("indices_input");
    mParams.fp16 = true;
    //init mEngine

}

bool CenterPoint::buildFromOnnx(const std::string& onnx_file, const std::string& engine_file, bool save) {
    auto startTime = std::chrono::high_resolution_clock::now();
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }


    auto constructed = constructNetwork(builder, network, config, parser, onnx_file);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }

    if(save)
    {
        std::ofstream engineFile(engine_file, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Cannot open engine file: " << engine_file << std::endl;
            return false;
        }

        SampleUniquePtr<IHostMemory> serializedEngine{mEngine->serialize()};
        if (serializedEngine == nullptr)
        {
            std::cout<< "Engine serialization failed" << std::endl;
            return false;
        }

        engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
        std::cout<< "Engine write: " <<engineFile.fail()<< serializedEngine->size()<<  std::endl;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double buildDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    sample::gLogInfo << "building  Time: " << buildDuration << " ms"<< std::endl;


    sample::gLogInfo << "getNbInputs: " << network->getNbInputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs: " << network->getNbOutputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs Name: " << network->getOutput(0)->getName() << " \n" << std::endl;




    return true;
}

bool CenterPoint::loadFromEngine(const std::string& engine_file) {
    auto startTime = std::chrono::high_resolution_clock::now();
    std::ifstream engineFile(engine_file, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << engine_file << std::endl;
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << engine_file << std::endl;
        return false;
    }

    SampleUniquePtr<nvinfer1::IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};

    std::cout << "deserialize  engine file size : " << fsize << std::endl;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), samplesCommon::InferDeleter());

    auto endTime = std::chrono::high_resolution_clock::now();
    double loadEngineDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    sample::gLogInfo << "loading  Time: " << loadEngineDuration << " ms"<< std::endl;
    return true;
}

bool CenterPoint::constructNetwork(CenterPoint::SampleUniquePtr<IBuilder> &builder,
                                   CenterPoint::SampleUniquePtr<INetworkDefinition> &network,
                                   CenterPoint::SampleUniquePtr<IBuilderConfig> &config,
                                   CenterPoint::SampleUniquePtr<nvonnxparser::IParser> &parser, std::string onnx_file) {

    auto parsed = parser->parseFromFile(onnx_file.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

std::vector<Box> CenterPoint::singleInference(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());


    if (!context)
    {
        std::cout << "inference context error!!" << std::endl;
    }

    float* hostPillars = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* hostIndex = static_cast<int32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));


    int pointNum = cloud->size();
    std::vector<Box> predResult;

    auto startTime = std::chrono::high_resolution_clock::now();
    preprocess(cloud, hostPillars, hostIndex, pointNum);
    auto endTime = std::chrono::high_resolution_clock::now();
    double preprocessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    startTime = std::chrono::high_resolution_clock::now();
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        std::cout << "inference context execute error!!" << std::endl;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    endTime = std::chrono::high_resolution_clock::now();

    double inferenceDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    startTime = std::chrono::high_resolution_clock::now();
    predResult.clear();
    postprocess(buffers, predResult);
    endTime = std::chrono::high_resolution_clock::now();
    double PostProcessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;

    sample::gLogInfo << "PreProcess Time: " << preprocessDuration << " ms"<< std::endl;
    sample::gLogInfo << "inferenceDuration Time: " << inferenceDuration << " ms"<< std::endl;
    sample::gLogInfo << "PostProcessDuration Time: " << PostProcessDuration << " ms"<< std::endl;

    return predResult;
}
