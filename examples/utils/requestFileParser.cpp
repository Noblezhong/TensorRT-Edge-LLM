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

#include "requestFileParser.h"

#include "common/checkMacros.h"
#include "common/inputLimits.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"

#include <cctype>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <cstring>

namespace trt_edgellm
{
namespace exampleUtils
{

using Json = nlohmann::json;

namespace
{
std::vector<unsigned char> decodeBase64(std::string const& input)
{
    static const std::array<int8_t, 256> kDecodeTable = []() {
        std::array<int8_t, 256> table{};
        table.fill(-1);
        for (int i = 0; i < 26; ++i)
        {
            table[static_cast<unsigned char>('A' + i)] = static_cast<int8_t>(i);
            table[static_cast<unsigned char>('a' + i)] = static_cast<int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i)
        {
            table[static_cast<unsigned char>('0' + i)] = static_cast<int8_t>(52 + i);
        }
        table[static_cast<unsigned char>('+')] = 62;
        table[static_cast<unsigned char>('/')] = 63;
        return table;
    }();

    std::vector<unsigned char> output;
    output.reserve(input.size() * 3 / 4);

    int val = 0;
    int valb = -8;
    for (unsigned char c : input)
    {
        if (std::isspace(c))
        {
            continue;
        }
        if (c == '=')
        {
            break;
        }
        int8_t decoded = kDecodeTable[c];
        if (decoded < 0)
        {
            throw std::runtime_error("Invalid base64 content in image payload");
        }
        val = (val << 6) + decoded;
        valb += 6;
        if (valb >= 0)
        {
            output.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return output;
}
} // namespace

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseRequestFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride, int64_t maxGenerateLengthOverride)
{
    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    check::check(inputFileStream.is_open(), "Failed to open input file: " + inputFilePath.string());
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            format::fmtstr("Failed to parse input file %s with error: %s", inputFilePath.string().c_str(), e.what()));
    }

    return parseRequestJson(std::move(inputData), batchSizeOverride, maxGenerateLengthOverride);
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseRequestJson(
    Json const& inputData, int32_t batchSizeOverride, int64_t maxGenerateLengthOverride)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    check::check(batchSize > 0, format::fmtstr("Invalid batch_size value: %d (must be positive)", batchSize));
    check::check(batchSize <= limits::security::kReasonableMaxBatchSize,
        format::fmtstr("Input rejected: batch_size %d exceeds limit %d. Limit defined in %s.", batchSize,
            limits::security::kReasonableMaxBatchSize, limits::kInputLimitsLocation));

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    check::check(maxGenerateLength > 0,
        format::fmtstr(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength)));

    bool applyChatTemplate = inputData.value("apply_chat_template", true);
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            check::check(loraPath.is_string(), "LoRA weight path for '" + loraName + "' must be a string");
            check::check(loraWeightsMap.find(loraName) == loraWeightsMap.end(),
                "Lora weights with name " + loraName + " already exists");
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    if (!(inputData.contains("requests") && inputData["requests"].is_array()))
    {
        throw std::runtime_error("'requests' array not found in input file");
    }

    auto& requestsArray = inputData["requests"];
    size_t const numRequests = requestsArray.size();

    for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
    {
        rt::LLMGenerationRequest batchRequest;
        batchRequest.temperature = temperature;
        batchRequest.topP = topP;
        batchRequest.topK = topK;
        batchRequest.maxGenerateLength = maxGenerateLength;
        batchRequest.applyChatTemplate = applyChatTemplate;
        batchRequest.addGenerationPrompt = addGenerationPrompt;
        batchRequest.enableThinking = enableThinking;

        std::string batchLoraWeightsName;
        bool firstInBatch = true;

        size_t const endIdx = std::min(startIdx + static_cast<size_t>(batchSize), numRequests);
        for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
        {
            auto const& requestItem = requestsArray[requestIdx];
            check::check(requestItem.is_object(), "Each request must be an object with 'messages' key");

            bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
            if (saveSystemPromptKVCache)
            {
                batchRequest.saveSystemPromptKVCache = true;
            }
            bool disableSpecDecode = requestItem.value("disable_spec_decode", false);
            if (disableSpecDecode)
            {
                batchRequest.disableSpecDecode = true;
            }

            check::check(requestItem.contains("messages") && requestItem["messages"].is_array(),
                "Each request object must contain a 'messages' array");

            auto const& messagesArray = requestItem["messages"];

            std::string requestLoraName;
            if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
            {
                requestLoraName = requestItem["lora_name"].get<std::string>();
                check::check(requestLoraName.empty() || loraWeightsMap.find(requestLoraName) != loraWeightsMap.end(),
                    "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
            }

            if (firstInBatch)
            {
                batchLoraWeightsName = requestLoraName;
                firstInBatch = false;
            }
            else
            {
                check::check(requestLoraName == batchLoraWeightsName,
                    "Different LoRA weights within the same batch are not supported");
            }

            std::vector<rt::Message> chatMessages;
            std::vector<rt::imageUtils::ImageData> imageBuffers;
            std::vector<rt::audioUtils::AudioData> audioBuffers;
            std::vector<int32_t> inputIds;

            check::check(messagesArray.size() <= limits::security::kMaxMessagesPerRequest,
                format::fmtstr("Input rejected: too many messages in request %zu: %zu (max: %zu). Limit defined in %s.",
                    requestIdx, messagesArray.size(), limits::security::kMaxMessagesPerRequest,
                    limits::kInputLimitsLocation));

            for (auto const& messageJson : messagesArray)
            {
                check::check(messageJson.contains("role") && messageJson.contains("content"),
                    "Each message must have 'role' and 'content' fields");

                rt::Message chatMsg;
                chatMsg.role = messageJson["role"].get<std::string>();

                auto const& contentJson = messageJson["content"];

                if (contentJson.is_string())
                {
                    std::string const& contentStr = contentJson.get<std::string>();
                    check::check(contentStr.size() <= limits::security::kMaxMessageContentSizeBytes,
                        format::fmtstr(
                            "Input rejected: message content too large in request %zu: %zu bytes (max: %zu). "
                            "Limit defined in %s.",
                            requestIdx, contentStr.size(), limits::security::kMaxMessageContentSizeBytes,
                            limits::kInputLimitsLocation));

                    rt::Message::MessageContent msgContent;
                    msgContent.type = "text";
                    msgContent.content = contentStr;
                    chatMsg.contents.push_back(msgContent);
                }
                else if (contentJson.is_array())
                {
                    check::check(contentJson.size() <= limits::security::kMaxContentItemsPerMessage,
                        format::fmtstr("Input rejected: too many content items in message %zu: %zu (max: %zu). "
                                       "Limit defined in %s.",
                            requestIdx, contentJson.size(), limits::security::kMaxContentItemsPerMessage,
                            limits::kInputLimitsLocation));

                    for (auto const& contentItemJson : contentJson)
                    {
                        check::check(contentItemJson.contains("type"), "Each content item must have a 'type' field");
                        rt::Message::MessageContent msgContent;
                        msgContent.type = contentItemJson["type"].get<std::string>();

                        if (msgContent.type == "text")
                        {
                            std::string const& textContent = contentItemJson["text"].get<std::string>();
                            check::check(textContent.size() <= limits::security::kMaxMessageContentSizeBytes,
                                format::fmtstr("Input rejected: message content too large in request %zu: %zu bytes "
                                               "(max: %zu). Limit defined in %s.",
                                    requestIdx, textContent.size(), limits::security::kMaxMessageContentSizeBytes,
                                    limits::kInputLimitsLocation));
                            msgContent.content = textContent;
                        }
                        else if (msgContent.type == "image")
                        {
                            if (contentItemJson.contains("image"))
                            {
                                msgContent.content = contentItemJson["image"].get<std::string>();
                                auto image = rt::imageUtils::loadImageFromFile(msgContent.content);
                                if (image.buffer != nullptr)
                                {
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else if (contentItemJson.contains("image_raw_base64"))
                            {
                                check::check(contentItemJson.contains("image_width") && contentItemJson.contains("image_height")
                                        && contentItemJson.contains("image_channels"),
                                    "Raw image payload must contain image_width/image_height/image_channels");
                                int64_t const width = contentItemJson["image_width"].get<int64_t>();
                                int64_t const height = contentItemJson["image_height"].get<int64_t>();
                                int64_t const channels = contentItemJson["image_channels"].get<int64_t>();
                                check::check(width > 0 && height > 0 && channels == 3,
                                    "Raw image payload must be RGB with positive width/height");
                                std::string const& imageBase64 = contentItemJson["image_raw_base64"].get<std::string>();
                                auto imageBytes = decodeBase64(imageBase64);
                                size_t const expectedSize = static_cast<size_t>(width * height * channels);
                                check::check(imageBytes.size() == expectedSize,
                                    format::fmtstr("Raw image payload size mismatch: got %zu, expected %zu",
                                        imageBytes.size(), expectedSize));
                                rt::Tensor imgTensor({height, width, channels}, rt::DeviceType::kCPU,
                                    nvinfer1::DataType::kUINT8, "requestFileParser::rawImageTensor");
                                std::memcpy(imgTensor.dataPointer<unsigned char>(), imageBytes.data(), expectedSize);
                                imageBuffers.emplace_back(std::move(imgTensor));
                                msgContent.content = contentItemJson.value("image_name", "raw_rgb_image");
                            }
                            else if (contentItemJson.contains("image_base64"))
                            {
                                std::string const& imageBase64 = contentItemJson["image_base64"].get<std::string>();
                                msgContent.content = contentItemJson.value("image_name", "socket_image");
                                auto imageBytes = decodeBase64(imageBase64);
                                auto image = rt::imageUtils::loadImageFromMemory(imageBytes.data(), imageBytes.size());
                                if (image.buffer != nullptr)
                                {
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "Image content must contain either 'image' path or 'image_base64' payload");
                            }
                        }
                        else if (msgContent.type == "audio")
                        {
                            msgContent.content = contentItemJson["audio"].get<std::string>();
                            rt::audioUtils::AudioData audio;
                            std::string const audioPath = msgContent.content;
                            size_t const dotPos = audioPath.find_last_of('.');
                            std::string const extension = (dotPos != std::string::npos) ? audioPath.substr(dotPos) : "";
                            if (extension == ".safetensors")
                            {
                                audio.melSpectrogramPath = audioPath;
                                audio.melSpectrogramFormat = "safetensors";
                                audioBuffers.push_back(std::move(audio));
                                LOG_INFO("Loaded mel-spectrogram from: %s (safetensors format)", audioPath.c_str());
                            }
                            else
                            {
                                LOG_WARNING(
                                    "Unsupported audio format: %s (only .safetensors mel-spectrograms supported)",
                                    audioPath.c_str());
                            }
                        }
                        else
                        {
                            throw std::runtime_error(format::fmtstr(
                                "Content type must be 'text', 'image', 'audio', but got: %s", msgContent.type.c_str()));
                        }
                        chatMsg.contents.push_back(msgContent);
                    }
                }
                else
                {
                    throw std::runtime_error("Message content must be a string or an array");
                }

                chatMessages.push_back(chatMsg);
            }

            if (requestItem.contains("input_ids"))
            {
                check::check(requestItem["input_ids"].is_array(), "request.input_ids must be an array of integers");
                inputIds.reserve(requestItem["input_ids"].size());
                for (auto const& tokenJson : requestItem["input_ids"])
                {
                    check::check(tokenJson.is_number_integer(), "request.input_ids must contain only integers");
                    inputIds.push_back(tokenJson.get<int32_t>());
                }
            }

            rt::LLMGenerationRequest::Request request;
            request.messages = std::move(chatMessages);
            request.imageBuffers = std::move(imageBuffers);
            request.audioBuffers = std::move(audioBuffers);
            request.inputIds = std::move(inputIds);
            batchRequest.requests.push_back(std::move(request));
        }

        if (!batchLoraWeightsName.empty())
        {
            batchRequest.loraWeightsName = batchLoraWeightsName;
        }

        batchedRequests.push_back(std::move(batchRequest));
    }

    return {std::move(loraWeightsMap), std::move(batchedRequests)};
}

} // namespace exampleUtils
} // namespace trt_edgellm
