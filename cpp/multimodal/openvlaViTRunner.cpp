/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openvlaViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "multimodal/imageUtils.h"
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{
namespace
{
std::string buildOpenVLARawPrompt(rt::LLMGenerationRequest::Request const& req)
{
    std::string prompt;
    bool hasAnyText = false;
    for (auto const& msg : req.messages)
    {
        for (auto const& content : msg.contents)
        {
            if (content.type == "text")
            {
                if (hasAnyText)
                {
                    prompt.push_back('\n');
                }
                prompt += content.content;
                hasAnyText = true;
            }
        }
    }
    return prompt;
}

std::string formatIdSlice(std::vector<int32_t> const& ids, size_t headCount = 12, size_t tailCount = 12)
{
    std::ostringstream oss;
    oss << "len=" << ids.size() << " head=[";
    size_t const headN = std::min(headCount, ids.size());
    for (size_t i = 0; i < headN; ++i)
    {
        if (i != 0)
        {
            oss << ", ";
        }
        oss << ids[i];
    }
    oss << "] tail=[";
    size_t const tailN = std::min(tailCount, ids.size());
    size_t const tailStart = ids.size() > tailN ? ids.size() - tailN : 0;
    for (size_t i = tailStart; i < ids.size(); ++i)
    {
        if (i != tailStart)
        {
            oss << ", ";
        }
        oss << ids[i];
    }
    oss << "]";
    return oss.str();
}
} // namespace

OpenVLAViTRunner::OpenVLAViTRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner(engineDir, stream)
{
    if (!validateAndFillConfig(engineDir))
    {
        throw std::runtime_error("OpenVLAViTRunner: invalid config");
    }
    if (!allocateBuffer(stream))
    {
        throw std::runtime_error("OpenVLAViTRunner: failed to allocate buffer");
    }
}

bool OpenVLAViTRunner::validateAndFillConfig(std::string const& engineDir)
{
    std::ifstream configFileStream(engineDir + "/config.json");
    if (!configFileStream.is_open())
    {
        LOG_ERROR("OpenVLAViTRunner: cannot open config.json");
        return false;
    }

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(configFileStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("OpenVLAViTRunner: parse config.json failed: %s", e.what());
        return false;
    }

    mModelType = multimodal::stringToModelType(jsonConfig.value("model_type", ""));
    if (mModelType != multimodal::ModelType::OPENVLA)
    {
        LOG_ERROR("OpenVLAViTRunner: invalid model_type");
        return false;
    }

    auto const& textCfg = jsonConfig.contains("text_config") ? jsonConfig["text_config"] : jsonConfig;
    mConfig.vocabSize = textCfg.value("vocab_size", 32064);
    if (jsonConfig.contains("image_token_id"))
    {
        mConfig.imageTokenId = jsonConfig["image_token_id"].get<int32_t>();
    }

    if (jsonConfig.contains("vision_config") && jsonConfig["vision_config"].contains("image_sizes"))
    {
        auto const& imageSizes = jsonConfig["vision_config"]["image_sizes"];
        if (!imageSizes.empty())
        {
            int64_t const img = imageSizes[0].get<int64_t>();
            mConfig.imageH = img;
            mConfig.imageW = img;
        }
    }

    // Optional preprocessor override for normalization.
    std::ifstream preprocFile(engineDir + "/preprocessor_config.json");
    if (preprocFile.is_open())
    {
        try
        {
            Json preproc = Json::parse(preprocFile);
            if (preproc.contains("means") && preproc["means"].is_array() && preproc["means"].size() >= 2)
            {
                for (int i = 0; i < 3; ++i)
                {
                    mConfig.meanA[i] = preproc["means"][0][i].get<float>();
                    mConfig.meanB[i] = preproc["means"][1][i].get<float>();
                }
            }
            if (preproc.contains("stds") && preproc["stds"].is_array() && preproc["stds"].size() >= 2)
            {
                for (int i = 0; i < 3; ++i)
                {
                    mConfig.stdA[i] = preproc["stds"][0][i].get<float>();
                    mConfig.stdB[i] = preproc["stds"][1][i].get<float>();
                }
            }
        }
        catch (...)
        {
            LOG_WARNING("OpenVLAViTRunner: failed to parse preprocessor_config.json, using defaults");
        }
    }

    nvinfer1::Dims const outShape = mVisualEngine->getTensorShape(binding_names::kVisualOutput);
    mConfig.outTokens = outShape.d[0];
    mConfig.outHiddenSize = outShape.d[1];
    return true;
}

bool OpenVLAViTRunner::allocateBuffer(cudaStream_t stream)
{
    bool setTensorAddressStatus{true};
    mVitInput = rt::Tensor({3, mConfig.channels, mConfig.imageH, mConfig.imageW}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "OpenVLAViTRunner::mVitInput");
    mVitInputHost = rt::Tensor({3, mConfig.channels, mConfig.imageH, mConfig.imageW}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kHALF, "OpenVLAViTRunner::mVitInputHost");
    mOutputEmbedding = rt::Tensor({mConfig.outTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "OpenVLAViTRunner::mOutputEmbedding");

    setTensorAddressStatus &= mVisualContext->setTensorAddress(binding_names::kVisualInput, mVitInput.rawPointer());
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kVisualOutput, mOutputEmbedding.rawPointer());
    if (!setTensorAddressStatus)
    {
        LOG_ERROR("OpenVLAViTRunner: set tensor address failed");
        return false;
    }

    CUDA_CHECK(cudaMemsetAsync(mVitInput.rawPointer(), 0, mVitInput.getMemoryCapacity(), stream));
    return true;
}

bool OpenVLAViTRunner::preprocessOneRequestImages(rt::LLMGenerationRequest::Request const& req, cudaStream_t stream)
{
    if (req.imageBuffers.size() != 3)
    {
        LOG_ERROR("OpenVLAViTRunner expects exactly 3 images, got %zu", req.imageBuffers.size());
        return false;
    }

    auto* dst = mVitInputHost.dataPointer<half>();
    int64_t const H = mConfig.imageH;
    int64_t const W = mConfig.imageW;
    int64_t const strideFrame = mConfig.channels * H * W;
    int64_t const strideChannel = H * W;

    for (int frame = 0; frame < 3; ++frame)
    {
        auto resizedA = rt::imageUtils::ImageData(rt::Tensor({H, W, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "openvla_resize_a"));
        auto resizedB = rt::imageUtils::ImageData(rt::Tensor({H, W, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "openvla_resize_b"));
        rt::imageUtils::resizeImage(
            req.imageBuffers[frame], resizedA, W, H, rt::imageUtils::InterpolationMode::kBICUBIC);
        rt::imageUtils::resizeImage(
            req.imageBuffers[frame], resizedB, W, H, rt::imageUtils::InterpolationMode::kBICUBIC);

        auto* srcA = resizedA.data();
        auto* srcB = resizedB.data();
        for (int64_t y = 0; y < H; ++y)
        {
            for (int64_t x = 0; x < W; ++x)
            {
                int64_t const srcIdx = (y * W + x) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    float const vA = static_cast<float>(srcA[srcIdx + c]) / 255.0F;
                    float const vB = static_cast<float>(srcB[srcIdx + c]) / 255.0F;
                    float const nA = (vA - mConfig.meanA[c]) / mConfig.stdA[c];
                    float const nB = (vB - mConfig.meanB[c]) / mConfig.stdB[c];
                    dst[frame * strideFrame + c * strideChannel + y * W + x] = __float2half(nA);
                    dst[frame * strideFrame + (c + 3) * strideChannel + y * W + x] = __float2half(nB);
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(
        mVitInput.rawPointer(), mVitInputHost.rawPointer(), mVitInputHost.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));

    if (std::getenv("OPENVLA_DUMP_VISION_INPUT") != nullptr)
    {
        char const* dumpPathEnv = std::getenv("OPENVLA_DUMP_VISION_INPUT_PATH");
        std::string dumpPath = dumpPathEnv != nullptr ? dumpPathEnv
                                                      : "/home/zt/tensorrt-edgellm-workspace/openvla_vision_input_trt.bin";
        std::ofstream out(dumpPath, std::ios::binary | std::ios::trunc);
        if (out.is_open())
        {
            out.write(reinterpret_cast<char const*>(mVitInputHost.rawPointer()),
                static_cast<std::streamsize>(mVitInputHost.getMemoryCapacity()));
            LOG_INFO("OpenVLA vision input dumped to %s (%zu bytes)", dumpPath.c_str(), mVitInputHost.getMemoryCapacity());
        }
        else
        {
            LOG_WARNING("OpenVLA vision input dump failed: cannot open %s", dumpPath.c_str());
        }
    }

    return true;
}

void OpenVLAViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer)
{
    int32_t nextImageTokenId = mConfig.vocabSize;
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;
        if (i < request.requests.size() && !request.requests[i].inputIds.empty())
        {
            ids = request.requests[i].inputIds;
        }
        else if (tokenizer != nullptr)
        {
            if (request.applyChatTemplate)
            {
                ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
            }
            else
            {
                ids = tokenizer->encode(buildOpenVLARawPrompt(request.requests[i]), true);
            }
        }
        if (ids.empty())
        {
            ids = {1};
        }

        LOG_INFO("OpenVLA raw text preprocess: batch=%zu applyChatTemplate=%d imageTokens=%ld promptIds(before)=%s",
            i, request.applyChatTemplate ? 1 : 0, static_cast<long>(mConfig.outTokens), formatIdSlice(ids).c_str());

        // Match OpenFly/OpenVLA Python predict_action(): ensure trailing token 29871
        // so generation starts from the same prompt boundary used during training/inference.
        if (ids.empty() || ids.back() != 29871)
        {
            ids.push_back(29871);
        }

        std::vector<int32_t> newIds;
        bool replacedPlaceholder = false;
        if (mConfig.imageTokenId >= 0)
        {
            for (size_t j = 0; j < ids.size(); ++j)
            {
                if (ids[j] == mConfig.imageTokenId)
                {
                    for (int64_t k = 0; k < mConfig.outTokens; ++k)
                    {
                        newIds.push_back(nextImageTokenId++);
                    }
                    replacedPlaceholder = true;
                }
                else
                {
                    newIds.push_back(ids[j]);
                }
            }
        }

        // Fallback for checkpoints/templates without explicit image placeholder token.
        if (!replacedPlaceholder)
        {
            newIds.clear();
            newIds.reserve(static_cast<size_t>(mConfig.outTokens) + ids.size());
            if (!ids.empty())
            {
                newIds.push_back(ids[0]);
                for (int64_t k = 0; k < mConfig.outTokens; ++k)
                {
                    newIds.push_back(nextImageTokenId++);
                }
                newIds.insert(newIds.end(), ids.begin() + 1, ids.end());
            }
            else
            {
                for (int64_t k = 0; k < mConfig.outTokens; ++k)
                {
                    newIds.push_back(nextImageTokenId++);
                }
            }
        }

        LOG_INFO("OpenVLA raw text preprocess: promptIds(after)=%s", formatIdSlice(newIds).c_str());
        batchedInputIds.emplace_back(std::move(newIds));
    }
}

bool OpenVLAViTRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream, bool imageOnly)
{
    check::check(request.requests.size() == 1, "OpenVLAViTRunner currently supports batch_size=1");
    if (!preprocessOneRequestImages(request.requests[0], stream))
    {
        return false;
    }
    if (!imageOnly)
    {
        textPreprocess(request, batchedInputIds, tokenizer);
    }
    mMultimodalMetrics.recordRun(3, mConfig.outTokens);
    return true;
}

bool OpenVLAViTRunner::infer(cudaStream_t stream) noexcept
{
    bool const ok = mVisualContext->enqueueV3(stream);
    if (!ok)
    {
        return false;
    }

    std::vector<half> hostOutput(static_cast<size_t>(mOutputEmbedding.getMemoryCapacity() / sizeof(half)));
    CUDA_CHECK(cudaMemcpyAsync(hostOutput.data(), mOutputEmbedding.rawPointer(), mOutputEmbedding.getMemoryCapacity(),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::getenv("OPENVLA_DUMP_VISUAL_OUTPUT") != nullptr)
    {
        char const* dumpPathEnv = std::getenv("OPENVLA_DUMP_VISUAL_OUTPUT_PATH");
        std::string dumpPath = dumpPathEnv != nullptr ? dumpPathEnv
                                                      : "/home/zt/tensorrt-edgellm-workspace/openvla_visual_output.bin";
        std::ofstream out(dumpPath, std::ios::binary | std::ios::trunc);
        if (out.is_open())
        {
            out.write(reinterpret_cast<char const*>(hostOutput.data()),
                static_cast<std::streamsize>(hostOutput.size() * sizeof(half)));
            LOG_INFO("OpenVLA visual output dumped to %s (%zu half values)", dumpPath.c_str(), hostOutput.size());
        }
        else
        {
            LOG_WARNING("OpenVLA visual output dump failed: cannot open %s", dumpPath.c_str());
        }
    }

    std::ostringstream oss;
    oss << "OpenVLA visual output head=[";
    size_t const n = std::min<size_t>(8, hostOutput.size());
    for (size_t i = 0; i < n; ++i)
    {
        if (i != 0)
        {
            oss << ", ";
        }
        oss << static_cast<float>(hostOutput[i]);
    }
    oss << "]";
    LOG_INFO("%s", oss.str().c_str());
    return true;
}

} // namespace rt
} // namespace trt_edgellm
