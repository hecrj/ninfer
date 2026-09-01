#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct OpenAIChatRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool include_usage          = false;
    bool output_tokens_explicit = false;
    // llama.cpp-compatible: attach cumulative timings to every streamed chunk instead of only
    // the terminal one.
    bool timings_per_token = false;
    // llama.cpp-compatible: emit prompt processing progress events while the prompt prefills.
    // Non-streaming responses already report exact prompt timings, so only streams change.
    bool return_progress = false;
};

OpenAIChatRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

struct OpenAIChatResponseIdentity {
    std::string id;
    std::string model;
    std::int64_t created = 0;
};

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model);
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome);

class OpenAIChatStream {
public:
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                     bool timings_per_token, bool return_progress = false);

    // Initial assistant-role chunk. It is emitted before Engine admission, when no prompt
    // accounting or timing data exists yet, so it never carries timings or prompt progress.
    std::string start();
    // Records the prompt accounting selected at admission. The Engine publishes it exactly once
    // before any output delta.
    void note_start(const ninfer::GenerationStart& start);
    // llama.cpp-compatible prompt progress chunk. The Engine publishes progress at admission and
    // after each completed prefill unit, always before the first output delta. With
    // timings_per_token the chunk also carries live prompt-side timings, like the output chunks.
    std::string prompt_progress(const ninfer::PromptProgress& progress);
    std::string reasoning_delta(const std::string& text, std::uint32_t committed_tokens);
    std::string content_delta(const std::string& text, std::uint32_t committed_tokens);
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    // Cumulative statistics observed by the HTTP layer while streaming. prompt/cached token
    // counts are exact from GenerationStart; committed output tokens accumulate from the
    // per-delta Engine counts; prompt_ms/predicted_ms are wall durations to the first delta and
    // since it. The terminal chunk always replaces them with the exact Engine timings.
    struct LiveTimings {
        bool started             = false;
        std::uint32_t prompt_tokens  = 0;
        std::uint32_t cached_tokens  = 0;
        std::uint64_t committed_tokens = 0;
        bool first_delta_seen    = false;
        std::chrono::steady_clock::time_point begin;
        std::chrono::steady_clock::time_point first_delta;
    };

    [[nodiscard]] nlohmann::json live_timings_json() const;
    // Live timings for prompt progress chunks. The prompt side tracks the progress event while
    // the decode side stays zero before the first output delta.
    [[nodiscard]] nlohmann::json
    progress_timings_json(const ninfer::PromptProgress& progress) const;
    void note_live_delta(std::uint32_t committed_tokens);

    OpenAIChatResponseIdentity identity_;
    std::string reasoning_;
    std::string content_;
    bool include_usage_      = false;
    bool timings_per_token_  = false;
    bool return_progress_    = false;
    bool started_            = false;
    bool content_started_    = false;
    bool finished_           = false;
    LiveTimings live_;
};

} // namespace ninfer::serve
