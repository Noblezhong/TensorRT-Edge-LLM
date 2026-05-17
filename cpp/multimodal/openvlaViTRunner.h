/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "multimodalRunner.h"
#include <array>

namespace trt_edgellm
{
namespace rt
{

struct OpenVLAViTConfig
{
    int64_t imageH{224};
    int64_t imageW{224};
    int64_t channels{6};
    int64_t outHiddenSize{0};
    int64_t outTokens{0};
    int32_t vocabSize{0};
    int32_t imageTokenId{-1};
    std::array<float, 3> meanA{{0.485F, 0.456F, 0.406F}};
    std::array<float, 3> stdA{{0.229F, 0.224F, 0.225F}};
    std::array<float, 3> meanB{{0.5F, 0.5F, 0.5F}};
    std::array<float, 3> stdB{{0.5F, 0.5F, 0.5F}};
};

class OpenVLAViTRunner : public MultimodalRunner
{
public:
    OpenVLAViTRunner(std::string const& engineDir, cudaStream_t stream);
    ~OpenVLAViTRunner() noexcept = default;

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream,
        bool imageOnly = false) override;

    bool infer(cudaStream_t stream) noexcept override;
    bool validateAndFillConfig(std::string const& engineDir) override;
    bool allocateBuffer(cudaStream_t stream) override;

private:
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer);
    bool preprocessOneRequestImages(rt::LLMGenerationRequest::Request const& req, cudaStream_t stream);

    OpenVLAViTConfig mConfig;
    rt::Tensor mVitInput{};
    rt::Tensor mVitInputHost{};
};

} // namespace rt
} // namespace trt_edgellm

