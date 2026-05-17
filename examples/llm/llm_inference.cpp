/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{
struct OpenFlyActionDecodeConfig
{
    bool enabled{false};
    int32_t vocabSize{0};
    int32_t nActionBins{256};
    int32_t actionDim{8};
    std::vector<float> q01;
    std::vector<float> q99;
    std::vector<bool> mask;
};

struct OpenFlyActionDecodeResult
{
    std::vector<int32_t> actionTokenIds;
    std::vector<int32_t> actionBins;
    std::vector<float> normalizedActions;
    std::vector<float> unnormalizedActions;
    int32_t actionId{-1};
};

static bool loadOpenFlyActionDecodeConfig(std::string const& multimodalEngineDir, OpenFlyActionDecodeConfig& outCfg)
{
    std::vector<std::string> candidatePaths;
    if (!multimodalEngineDir.empty())
    {
        candidatePaths.push_back(multimodalEngineDir + "/config.json");
        candidatePaths.push_back(multimodalEngineDir + "/visual/config.json");
    }

    for (auto const& path : candidatePaths)
    {
        std::ifstream in(path);
        if (!in.is_open())
        {
            continue;
        }

        Json cfg;
        try
        {
            cfg = Json::parse(in);
        }
        catch (...)
        {
            continue;
        }

        if (cfg.value("model_type", "") != "openvla")
        {
            continue;
        }

        auto const& textCfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;
        auto const& visionCfg = cfg.contains("vision_config") ? cfg["vision_config"] : cfg;
        auto const& normStats = visionCfg.contains("norm_stats") ? visionCfg["norm_stats"] : Json{};
        if (!normStats.is_object() || !normStats.contains("vlnv1"))
        {
            continue;
        }

        auto const& actionStats = normStats["vlnv1"]["action"];
        if (!actionStats.is_object() || !actionStats.contains("q01") || !actionStats.contains("q99"))
        {
            continue;
        }

        outCfg.enabled = true;
        outCfg.vocabSize = textCfg.value("vocab_size", 0);
        outCfg.nActionBins = visionCfg.value("n_action_bins", 256);
        outCfg.actionDim = static_cast<int32_t>(actionStats["q01"].size());
        outCfg.q01.clear();
        outCfg.q99.clear();
        outCfg.mask.clear();
        for (auto const& v : actionStats["q01"])
        {
            outCfg.q01.push_back(v.get<float>());
        }
        for (auto const& v : actionStats["q99"])
        {
            outCfg.q99.push_back(v.get<float>());
        }
        if (actionStats.contains("mask"))
        {
            for (auto const& v : actionStats["mask"])
            {
                outCfg.mask.push_back(v.get<bool>());
            }
        }
        else
        {
            outCfg.mask.assign(outCfg.q01.size(), true);
        }
        return true;
    }

    return false;
}

static int32_t convertOpenFlyActionToId(std::vector<float> const& action)
{
    static constexpr float kTemplates[10][8] = {
        {1, 0, 0, 0, 0, 0, 0, 0},  // stop
        {0, 3, 0, 0, 0, 0, 0, 0},  // move forward
        {0, 0, 15, 0, 0, 0, 0, 0}, // turn left 30
        {0, 0, 0, 15, 0, 0, 0, 0}, // turn right 30
        {0, 0, 0, 0, 2, 0, 0, 0},  // go up
        {0, 0, 0, 0, 0, 2, 0, 0},  // go down
        {0, 0, 0, 0, 0, 0, 5, 0},  // move left
        {0, 0, 0, 0, 0, 0, 0, 5},  // move right
        {0, 6, 0, 0, 0, 0, 0, 0},  // move forward 6
        {0, 9, 0, 0, 0, 0, 0, 0},  // move forward 9
    };

    if (action.size() != 8)
    {
        return -1;
    }

    for (int actionId = 0; actionId < 10; ++actionId)
    {
        bool matched = true;
        for (int i = 0; i < 8; ++i)
        {
            if (static_cast<int32_t>(std::llround(action[i])) != static_cast<int32_t>(kTemplates[actionId][i]))
            {
                matched = false;
                break;
            }
        }
        if (matched)
        {
            return actionId;
        }
    }
    return 0;
}

static bool decodeOpenFlyAction(
    std::vector<int32_t> const& outputIds, OpenFlyActionDecodeConfig const& cfg, OpenFlyActionDecodeResult& out)
{
    if (!cfg.enabled || cfg.actionDim <= 0 || cfg.vocabSize <= 0)
    {
        return false;
    }
    if (static_cast<int32_t>(outputIds.size()) < cfg.actionDim)
    {
        return false;
    }

    // `outputIds` stores the generated response tokens only for OpenFly requests.
    // The action tokens are the first `actionDim` generated tokens, followed by an optional EOS.
    out.actionTokenIds.assign(outputIds.begin(), outputIds.begin() + cfg.actionDim);
    out.actionBins.resize(cfg.actionDim);
    out.normalizedActions.resize(cfg.actionDim);
    out.unnormalizedActions.resize(cfg.actionDim);

    int32_t const binCount = std::max(1, cfg.nActionBins - 1);
    for (int32_t i = 0; i < cfg.actionDim; ++i)
    {
        int32_t const tokenId = out.actionTokenIds[i];
        int32_t discretized = cfg.vocabSize - tokenId;
        discretized = std::clamp(discretized - 1, 0, binCount - 1);
        out.actionBins[i] = discretized;

        float const normalized = -1.0F + (2.0F * (static_cast<float>(discretized) + 0.5F))
                                               / static_cast<float>(binCount);
        out.normalizedActions[i] = normalized;

        float const q01 = i < static_cast<int32_t>(cfg.q01.size()) ? cfg.q01[i] : 0.0F;
        float const q99 = i < static_cast<int32_t>(cfg.q99.size()) ? cfg.q99[i] : 0.0F;
        bool const mask = i < static_cast<int32_t>(cfg.mask.size()) ? cfg.mask[i] : true;
        float const action = mask ? 0.5F * (normalized + 1.0F) * (q99 - q01) + q01 : normalized;
        out.unnormalizedActions[i] = action;
    }

    out.actionId = convertOpenFlyActionToId(out.unnormalizedActions);
    return true;
}
} // namespace

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    EAGLE = 910,
    EAGLE_DRAFT_TOP_K = 911,
    EAGLE_DRAFT_STEP = 912,
    EAGLE_VERIFY_TREE_SIZE = 913,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915
};

// Struct to hold Eagle-specific arguments for speculative decoding
struct EagleArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // This controls the branching factor at each level of the draft tree.
    int32_t draftTopK{10};

    // Number of drafting steps to perform with the draft model.
    // Each step extends the draft tree by one more level.
    int32_t draftStep{6};

    // Number of tokens to select from the complete draft tree for base model verification.
    // The total draft tree size is: 1 + draftTopK + (draftStep - 1) * draftTopK * draftTopK
    // This parameter should be <= total draft tree size for optimal performance.
    int32_t verifyTreeSize{60};
};

struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    // Override parameters (only batchSize and maxGenerateLength can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    EagleArgs eagleArgs;
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--eagle] "
                 "[--eagleDraftTopK=<number>] [--eagleDraftStep=<number>] "
                 "[--eagleVerifyTreeSize=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr << "  --eagle                   Enable Eagle speculative decoding mode" << std::endl;
    std::cerr << "  --eagleDraftTopK          Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            Controls branching factor at each draft tree level" << std::endl;
    std::cerr << "  --eagleDraftStep          Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the draft tree by one more level" << std::endl;
    std::cerr << "  --eagleVerifyTreeSize     Number of tokens for base model verification (default: 60)" << std::endl;
    std::cerr << "                            Total draft tree size: 1 + topK + (step-1) * topK^2" << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, LLMInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"eagle", no_argument, 0, LLMInferenceOptionId::EAGLE},
        {"eagleDraftTopK", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_TOP_K},
        {"eagleDraftStep", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_STEP},
        {"eagleVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case LLMInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case LLMInferenceOptionId::DEBUG: args.debug = true; break;
        case LLMInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case LLMInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case LLMInferenceOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case LLMInferenceOptionId::EAGLE: args.eagleArgs.enabled = true; break;
        case LLMInferenceOptionId::EAGLE_DRAFT_TOP_K:
            try
            {
                args.eagleArgs.draftTopK = std::stoi(optarg);
                if (args.eagleArgs.draftTopK <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftTopK value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftTopK value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_DRAFT_STEP:
            try
            {
                args.eagleArgs.draftStep = std::stoi(optarg);
                if (args.eagleArgs.draftStep <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftStep value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftStep value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE:
            try
            {
                args.eagleArgs.verifyTreeSize = std::stoi(optarg);
                if (args.eagleArgs.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid eagleVerifyTreeSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleVerifyTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength <= 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        default: return false;
        }
    }

    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
    }

    if (args.eagleArgs.enabled)
    {
        LOG_INFO("Eagle mode enabled");
        LOG_INFO("Eagle draft topK: %d", args.eagleArgs.draftTopK);
        LOG_INFO("Eagle draft step: %d", args.eagleArgs.draftStep);
        LOG_INFO("Eagle verify tree size: %d", args.eagleArgs.verifyTreeSize);
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

// Thin wrapper around the shared parser in examples/utils/requestFileParser.h.
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    return exampleUtils::parseRequestFile(inputFilePath, batchSizeOverride, maxGenerateLengthOverride);
}

int main(int argc, char* argv[])
{
    NVTX_SCOPED_RANGE(nvtx_main, "llm_inference");
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    bool profilerEnabled = args.dumpProfile;
    MemoryMonitor memoryMonitor;
    OpenFlyActionDecodeConfig openFlyDecodeConfig;
    if (!args.multimodalEngineDir.empty())
    {
        (void) loadOpenFlyActionDecodeConfig(args.multimodalEngineDir, openFlyDecodeConfig);
    }
    // Start memory monitoring at the beginning if profiling is enabled
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    auto pluginHandles = loadEdgellmPluginLib();
    // load input file and parse to requests
    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    try
    {
        std::tie(loraWeightsMap, batchedRequests)
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    // Create unified runtime (handles both vanilla and Eagle spec-decode modes)
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> runtime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (args.eagleArgs.enabled)
    {
        rt::EagleDraftingConfig draftingConfig{
            args.eagleArgs.draftTopK, args.eagleArgs.draftStep, args.eagleArgs.verifyTreeSize};
        try
        {
            runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, draftingConfig, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime with Eagle spec-decode: %s", e.what());
            return EXIT_FAILURE;
        }
    }
    else
    {
        // Standard vanilla-only mode (no draft model)
        try
        {
            runtime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize runtime: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        LOG_WARNING("Failed to capture CUDA graph for decoding, proceeding with normal engine execution.");
    }

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        // Disable profiling for warmup runs
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = batchedRequests[0];

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = runtime->handleRequest(firstRequest, warmupResponse, stream);

            if (!requestStatus)
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(true);
    }

    // Structure to collect all responses for JSON export
    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;

    // Process each request with progress indication
    LOG_INFO("Processing %zu batched requests...", batchedRequests.size());
    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        auto& request = batchedRequests[requestIdx];
        rt::LLMGenerationResponse response;

        // Show progress every 10% or every 100 requests, whichever is smaller
        size_t progressInterval = std::max(size_t(1), std::min(batchedRequests.size() / 10, size_t(100)));
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == batchedRequests.size() - 1)
        {
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, batchedRequests.size(),
                100.0 * (requestIdx + 1) / batchedRequests.size());
        }

        bool requestStatus = runtime->handleRequest(request, response, stream);

        if (requestStatus)
        {
            // Display inference output to console if --dumpOutput is enabled
            if (args.dumpOutput)
            {
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                        response.outputTexts[batchIdx].c_str());
                }
            }
        }
        else
        {
            // Handle failed request - highlight failures
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // Add to JSON output with UTF-8 validation on output text
        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            bool const hasOutputText = requestStatus && batchIdx < response.outputTexts.size();
            bool const hasOutputIds = requestStatus && batchIdx < response.outputIds.size();
            std::string outputText = hasOutputText ? response.outputTexts[batchIdx] : errorMessage;
            auto const* formattedRequest
                = batchIdx < request.formattedRequests.size() ? &request.formattedRequests[batchIdx] : nullptr;
            // Validate UTF-8 for output text (inputs are always valid)
            // If invalid UTF-8 detected, error message is returned and original text is logged
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["output_ids"] = hasOutputIds ? Json(response.outputIds[batchIdx]) : Json::array();
            responseJson["request_idx"] = requestIdx;
            responseJson["batch_idx"] = batchIdx;
            // Store messages for reference
            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : request.requests[batchIdx].messages)
            {
                nlohmann::json msgJson;
                msgJson["role"] = msg.role;
                msgJson["content"] = nlohmann::json::array();
                for (auto const& content : msg.contents)
                {
                    nlohmann::json contentJson;
                    contentJson["type"] = content.type;
                    if (content.type == "text")
                    {
                        contentJson["text"] = content.content;
                    }
                    else if (content.type == "image")
                    {
                        contentJson["image"] = content.content;
                    }
                    else if (content.type == "video")
                    {
                        contentJson["video"] = content.content;
                    }
                    msgJson["content"].push_back(contentJson);
                }
                messagesJson.push_back(msgJson);
            }
            responseJson["messages"] = messagesJson;
            // Store formatted prompts for reference
            responseJson["formatted_system_prompt"] = formattedRequest ? formattedRequest->formattedSystemPrompt : "";
            responseJson["formatted_complete_request"]
                = formattedRequest ? formattedRequest->formattedCompleteRequest : "";

            if (openFlyDecodeConfig.enabled && hasOutputIds)
            {
                OpenFlyActionDecodeResult actionResult;
                if (decodeOpenFlyAction(response.outputIds[batchIdx], openFlyDecodeConfig, actionResult))
                {
                    responseJson["openfly_action_token_ids"] = actionResult.actionTokenIds;
                    responseJson["openfly_action_bins"] = actionResult.actionBins;
                    responseJson["openfly_normalized_action"] = actionResult.normalizedActions;
                    responseJson["openfly_action"] = actionResult.unnormalizedActions;
                    responseJson["openfly_action_id"] = actionResult.actionId;
                }
            }
            outputData["responses"].push_back(responseJson);
        }
    }

    // Final processing summary
    LOG_INFO("Processing complete: %zu/%zu batched requests successful", batchedRequests.size() - failedCount,
        batchedRequests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    if (profilerEnabled)
    {
        // Stop memory monitoring for examples
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Performance Summary ===" << std::endl;
        auto prefillMetrics = runtime->getPrefillMetrics();
        auto multimodalMetrics = runtime->getMultimodalMetrics();
        outputPrefillProfile(profileOutput, prefillMetrics);
        if (args.eagleArgs.enabled)
        {
            auto eagleGenerationMetrics = runtime->getEagleGenerationMetrics();
            outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
        }
        else
        {
            outputGenerationProfile(profileOutput, runtime->getGenerationMetrics());
        }
        outputMultimodalProfile(profileOutput, multimodalMetrics);
        outputMemoryProfile(profileOutput, memoryMonitor);
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // Export profile to JSON file
    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;

            // Add high-level metrics from unified runtime
            addJsonPrefillSummary(profileJson, runtime->getPrefillMetrics());
            if (args.eagleArgs.enabled)
            {
                addJsonEagleGenerationSummary(profileJson, runtime->getEagleGenerationMetrics());
            }
            else
            {
                addJsonGenerationSummary(profileJson, runtime->getGenerationMetrics());
            }
            addJsonMultimodalSummary(profileJson, runtime->getMultimodalMetrics());

            // Add detailed timing stages
            addJsonTimingStages(profileJson);

            // Add memory usage
            addJsonMemorySummary(profileJson, memoryMonitor);

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2); // Pretty print with 2 space indentation
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    // Export to JSON file
    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4); // Pretty print with 4 spaces indentation
            outputFile.close();
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    // Return false if any request failed
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
