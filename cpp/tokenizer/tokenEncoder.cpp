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

#include "tokenEncoder.h"
#include "common/inputLimits.h"
#include "tokenizerUtils.h"
#include <cstdio>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace tokenizer
{

namespace
{
std::string mergeKey(std::string const& left, std::string const& right)
{
    return left + "\x1f" + right;
}
} // namespace

// Size limits for token encoder processing
constexpr size_t LARGE_PIECE_WARNING_BYTES = 65536; // 64KB warning threshold

TokenEncoder::TokenEncoder(Type type, bool byteFallback) noexcept
    : mType(type)
    , mByteFallback(byteFallback)
    , mVocabSize(0)
{
}

bool TokenEncoder::initialize(
    TokenToRanks const& vocab, TokenToRanks const& specialTokens, TokenToRanks const& mergeRanks)
{
    if (vocab.empty())
    {
        return false;
    }

    mEncoder = vocab;
    mSpecialTokensEncoder = specialTokens;
    mMergeRanks = mergeRanks;

    // Build reverse mappings using utility function
    mDecoder = reverseEncoder(mEncoder);
    mSpecialTokensDecoder = reverseEncoder(mSpecialTokensEncoder);

    // Calculate vocab size as total number of tokens
    mVocabSize = mEncoder.size() + mSpecialTokensEncoder.size();
    return true;
}

bool TokenEncoder::encode(std::string const& piece, std::vector<Rank>& output) const noexcept
{
    if (piece.empty())
    {
        return true;
    }

    if (piece.size() > limits::tokenizer::kMaxTokenPieceSizeBytes)
    {
        LOG_ERROR("Input text piece too large: %zu bytes", piece.size());
        return false;
    }

    if (piece.size() > LARGE_PIECE_WARNING_BYTES) // 64KB warning per piece
    {
        LOG_WARNING("Very large piece encountered: %zu bytes", piece.size());
    }

    try
    {
        switch (mType)
        {
        case BPE:
            if (!bytePairEncode(piece, output))
            {
                if (mByteFallback && byteFallbackEncode(piece, output))
                {
                    return true;
                }
                return false;
            }
            break;
        default: LOG_ERROR("Unknown or unsupported encoder type: %s", getTypeString(mType).c_str()); return false;
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("TokenEncoder::encode failed on piece: %s", piece.c_str());
        return false;
    }
}

bool TokenEncoder::decode(std::vector<Rank> const& tokens, std::string& output, bool skipSpecialTokens) const noexcept
{
    try
    {
        output.clear();
        output.reserve(tokens.size() * 4); // Rough estimate

        for (Rank token : tokens)
        {
            auto it = mDecoder.find(token);
            if (it != mDecoder.end())
            {
                output += it->second;
            }
            else if (!skipSpecialTokens)
            {
                auto specialIt = mSpecialTokensDecoder.find(token);
                if (specialIt != mSpecialTokensDecoder.end())
                {
                    output += specialIt->second;
                }
                else
                {
                    LOG_ERROR("Unknown token %d during decode", token);
                    return false;
                }
            }
            // Skip unknown tokens if skipSpecialTokens is true
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("TokenEncoder::decode failed: %s", e.what());
        return false;
    }
}

bool TokenEncoder::hasToken(std::string const& token) const noexcept
{
    return mEncoder.find(token) != mEncoder.end() || mSpecialTokensEncoder.find(token) != mSpecialTokensEncoder.end();
}

Rank TokenEncoder::getTokenRank(std::string const& token) const noexcept
{
    auto it = mEncoder.find(token);
    if (it != mEncoder.end())
    {
        return it->second;
    }

    auto specialIt = mSpecialTokensEncoder.find(token);
    if (specialIt != mSpecialTokensEncoder.end())
    {
        return specialIt->second;
    }

    return -1; // Token not found
}

std::string TokenEncoder::getRankToken(Rank rank) const
{
    auto it = mDecoder.find(rank);
    if (it != mDecoder.end())
    {
        return it->second;
    }

    auto specialIt = mSpecialTokensDecoder.find(rank);
    if (specialIt != mSpecialTokensDecoder.end())
    {
        return specialIt->second;
    }

    return ""; // Rank not found
}

bool TokenEncoder::bytePairEncode(std::string const& piece, std::vector<Rank>& output) const
{
    if (piece.empty())
    {
        return true;
    }

    // Check if the piece is already in vocabulary
    auto it = mEncoder.find(piece);
    if (it != mEncoder.end())
    {
        output.emplace_back(it->second);
        return true;
    }

    std::vector<std::string> symbols;
    auto const cpts = unicodeCptsFromUtf8(piece);
    symbols.reserve(cpts.size());
    for (auto const cpt : cpts)
    {
        symbols.emplace_back(unicodeCptToUtf8(cpt));
    }

    if (symbols.empty())
    {
        return true;
    }

    auto findBestMerge = [&]() -> std::pair<size_t, Rank> {
        auto const MAX_RANK = std::numeric_limits<Rank>::max();
        std::pair<size_t, Rank> best{std::numeric_limits<size_t>::max(), MAX_RANK};
        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            auto itMerge = mMergeRanks.find(mergeKey(symbols[i], symbols[i + 1]));
            if (itMerge != mMergeRanks.end() && itMerge->second < best.second)
            {
                best = {i, itMerge->second};
            }
        }
        return best;
    };

    while (true)
    {
        auto best = findBestMerge();
        if (best.second == std::numeric_limits<Rank>::max())
        {
            break;
        }

        size_t const idx = best.first;
        symbols[idx] += symbols[idx + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(idx + 1));
    }

    for (auto const& token : symbols)
    {
        auto tokenIt = mEncoder.find(token);
        if (tokenIt != mEncoder.end())
        {
            output.emplace_back(tokenIt->second);
            continue;
        }

        if (mByteFallback)
        {
            bool fallbackOk = true;
            char tokenBuf[8];
            for (unsigned char byte : token)
            {
                std::snprintf(tokenBuf, sizeof(tokenBuf), "<0x%02X>", static_cast<unsigned int>(byte));
                auto byteIt = mEncoder.find(tokenBuf);
                if (byteIt == mEncoder.end())
                {
                    fallbackOk = false;
                    break;
                }
                output.emplace_back(byteIt->second);
            }
            if (fallbackOk)
            {
                continue;
            }
        }

        LOG_ERROR("Token not found in encoder during bytePairEncode: '%s'", token.c_str());
        return false;
    }

    return true;
}

bool TokenEncoder::byteFallbackEncode(std::string const& piece, std::vector<Rank>& output) const
{
    if (piece.empty())
    {
        return true;
    }

    std::vector<Rank> fallbackTokens;
    fallbackTokens.reserve(piece.size());

    char tokenBuf[8];
    for (unsigned char byte : piece)
    {
        std::snprintf(tokenBuf, sizeof(tokenBuf), "<0x%02X>", static_cast<unsigned int>(byte));
        auto it = mEncoder.find(tokenBuf);
        if (it == mEncoder.end())
        {
            LOG_ERROR("Byte fallback token not found in encoder: '%s'", tokenBuf);
            return false;
        }
        fallbackTokens.emplace_back(it->second);
    }

    output.insert(output.end(), fallbackTokens.begin(), fallbackTokens.end());
    return true;
}

std::string TokenEncoder::getTypeString(Type type) const
{
    switch (type)
    {
    case BPE: return "BPE";
    default: return "UNKNOWN";
    }
}

} // namespace tokenizer
} // namespace trt_edgellm
