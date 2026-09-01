#include "serve/openai_chat.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int cached_tokens     = 0;
    int reasoning_tokens  = 0;
};

const char* finish_reason(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "length";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
    case ninfer::FinishReason::Cancelled:
        return "stop";
    }
    return "stop";
}

std::vector<ToolCall>
materialize_tool_calls(const std::vector<ninfer::GeneratedToolCall>& generated) {
    std::vector<ToolCall> calls;
    calls.reserve(generated.size());
    for (const ninfer::GeneratedToolCall& call : generated) {
        calls.push_back(ToolCall{.id             = new_openai_chat_tool_call_id(),
                                 .name           = call.name,
                                 .arguments_json = call.arguments_json});
    }
    return calls;
}

Json tool_calls_json(const std::vector<ToolCall>& calls, bool include_index) {
    Json output = Json::array();
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const ToolCall& call = calls[index];
        Json value           = {{"id", call.id},
                                {"type", "function"},
                                {"function", Json{{"name", call.name}, {"arguments", call.arguments_json}}}};
        if (include_index) { value["index"] = static_cast<int>(index); }
        output.push_back(std::move(value));
    }
    return output;
}

Json usage_json(const CompletionUsage& usage) {
    const int cached_tokens = std::clamp(usage.cached_tokens, 0, usage.prompt_tokens);
    return Json{{"prompt_tokens", usage.prompt_tokens},
                {"prompt_tokens_details", Json{{"cached_tokens", cached_tokens}}},
                {"completion_tokens", usage.completion_tokens},
                {"completion_tokens_details",
                 Json{{"reasoning_tokens", std::max(0, usage.reasoning_tokens)}}},
                {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
}

// llama.cpp-compatible prompt processing progress: total admitted prompt tokens, tokens reused
// from the prefix cache, the prompt position reached (cached + computed so far), and wall
// milliseconds since the request was admitted.
Json prompt_progress_json(const ninfer::PromptProgress& progress) {
    return Json{{"total", progress.total_tokens},
                {"cache", progress.cached_tokens},
                {"processed", progress.processed_tokens},
                {"time_ms", progress.elapsed_ms}};
}

CompletionUsage usage_from(const GenerationOutcome& outcome) {
    return CompletionUsage{
        .prompt_tokens     = outcome.prompt_tokens,
        .completion_tokens = outcome.completion_tokens,
        .cached_tokens     = static_cast<int>(outcome.metrics.prefix_cache_hit_tokens),
        .reasoning_tokens  = outcome.reasoning_tokens,
    };
}

Json base_payload(const OpenAIChatResponseIdentity& identity, const char* object) {
    return Json{{"id", identity.id},
                {"object", object},
                {"created", identity.created},
                {"model", identity.model}};
}

Json stream_choice(Json delta, Json finish_reason = nullptr) {
    return Json{{"index", 0},
                {"delta", std::move(delta)},
                {"logprobs", nullptr},
                {"finish_reason", std::move(finish_reason)}};
}

std::string event(Json payload) { return "data: " + payload.dump() + "\n\n"; }

// llama.cpp-compatible completion timings. Field names, units (milliseconds, tokens per
// second), and zero-divisor behavior mirror the llama.cpp server slot-stats contract:
// - cache_n/prompt_n split the prompt into tokens reused from the prefix cache and tokens this
//   request actually computed;
// - prompt_ms/predicted_ms are the prefill and decode program durations;
// - the first output token is produced by the prefill batch, so decode rates divide by
//   predicted_n - 1 decode steps;
// - every rate is 0.0 while its divisor is not positive;
// - draft fields appear only when speculative decoding drafted tokens.
struct CompletionTimings {
    std::uint64_t cache_n          = 0;
    std::uint64_t prompt_n         = 0;
    double prompt_ms               = 0.0;
    double prompt_per_token_ms     = 0.0;
    double prompt_per_second       = 0.0;
    std::uint64_t predicted_n      = 0;
    double predicted_ms            = 0.0;
    double predicted_per_token_ms  = 0.0;
    double predicted_per_second    = 0.0;
    std::uint64_t draft_n          = 0;
    std::uint64_t draft_n_accepted = 0;
};

double ms_per_token(std::uint64_t tokens, double milliseconds) {
    return tokens > 0 ? milliseconds / static_cast<double>(tokens) : 0.0;
}

double tokens_per_second(std::uint64_t tokens, double milliseconds) {
    return milliseconds > 0.0 ? 1.0e3 * static_cast<double>(tokens) / milliseconds : 0.0;
}

CompletionTimings make_completion_timings(std::uint32_t prompt_tokens, std::uint32_t cached_tokens,
                                          std::uint32_t completion_tokens, double prompt_ms,
                                          double predicted_ms, std::uint64_t draft_tokens,
                                          std::uint64_t draft_accepted) {
    CompletionTimings timings;
    timings.cache_n = std::min<std::uint64_t>(cached_tokens, prompt_tokens);
    timings.prompt_n = prompt_tokens - static_cast<std::uint32_t>(timings.cache_n);
    timings.prompt_ms = prompt_ms;
    timings.prompt_per_token_ms = ms_per_token(timings.prompt_n, timings.prompt_ms);
    timings.prompt_per_second   = tokens_per_second(timings.prompt_n, timings.prompt_ms);
    timings.predicted_n = completion_tokens;
    timings.predicted_ms = predicted_ms;
    const std::uint64_t decode_steps = completion_tokens > 0 ? completion_tokens - 1 : 0;
    timings.predicted_per_token_ms = ms_per_token(decode_steps, timings.predicted_ms);
    timings.predicted_per_second   = tokens_per_second(decode_steps, timings.predicted_ms);
    timings.draft_n          = draft_tokens;
    timings.draft_n_accepted = draft_accepted;
    return timings;
}

Json timings_json(const CompletionTimings& timings) {
    Json output = {
        {"cache_n", timings.cache_n},
        {"prompt_n", timings.prompt_n},
        {"prompt_ms", timings.prompt_ms},
        {"prompt_per_token_ms", timings.prompt_per_token_ms},
        {"prompt_per_second", timings.prompt_per_second},
        {"predicted_n", timings.predicted_n},
        {"predicted_ms", timings.predicted_ms},
        {"predicted_per_token_ms", timings.predicted_per_token_ms},
        {"predicted_per_second", timings.predicted_per_second},
    };
    if (timings.draft_n > 0) {
        output["draft_n"]          = timings.draft_n;
        output["draft_n_accepted"] = timings.draft_n_accepted;
    }
    return output;
}

CompletionTimings outcome_timings(const GenerationOutcome& outcome) {
    return make_completion_timings(
        static_cast<std::uint32_t>(std::max(0, outcome.prompt_tokens)),
        outcome.metrics.prefix_cache_hit_tokens,
        static_cast<std::uint32_t>(std::max(0, outcome.completion_tokens)),
        std::max(0.0, outcome.metrics.prefill_seconds) * 1000.0,
        std::max(0.0, outcome.metrics.decode_seconds) * 1000.0,
        outcome.metrics.speculative_draft_tokens,
        outcome.metrics.speculative_accepted_tokens);
}

std::string chunk(const OpenAIChatResponseIdentity& identity, Json delta, Json finish_reason,
                  bool include_usage, Json timings = nullptr) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array({stream_choice(std::move(delta), std::move(finish_reason))});
    if (include_usage) { payload["usage"] = nullptr; }
    if (!timings.is_null()) { payload["timings"] = std::move(timings); }
    return event(std::move(payload));
}

std::string usage_chunk(const OpenAIChatResponseIdentity& identity, const CompletionUsage& usage,
                        Json timings) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array();
    payload["usage"]   = usage_json(usage);
    if (!timings.is_null()) { payload["timings"] = std::move(timings); }
    return event(std::move(payload));
}

void require_prefix(std::string_view complete, std::string_view streamed, const char* channel) {
    if (!complete.starts_with(streamed)) {
        throw std::logic_error(std::string("streamed ") + channel +
                               " does not match terminal output");
    }
}

} // namespace

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model) {
    return OpenAIChatResponseIdentity{
        .id      = new_openai_chat_completion_id(),
        .model   = std::move(model),
        .created = unix_time_now(),
    };
}

std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome) {
    Json message = {{"role", "assistant"}, {"content", outcome.text}, {"refusal", nullptr}};
    const bool has_tool_calls = !outcome.tool_calls.empty();
    // vLLM/SGLang-compatible reasoning_content preserves the Engine's Reasoning/Content split.
    if (!outcome.reasoning.empty()) { message["reasoning_content"] = outcome.reasoning; }
    if (has_tool_calls) {
        const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
        message["content"]    = outcome.text.empty() ? Json(nullptr) : Json(outcome.text);
        message["tool_calls"] = tool_calls_json(calls, false);
    }

    Json payload       = base_payload(identity, "chat.completion");
    payload["choices"] = Json::array(
        {Json{{"index", 0},
              {"message", std::move(message)},
              {"logprobs", nullptr},
              {"finish_reason",
               has_tool_calls ? Json("tool_calls") : Json(finish_reason(outcome.finish_reason))}}});
    payload["usage"]   = usage_json(usage_from(outcome));
    payload["timings"] = timings_json(outcome_timings(outcome));
    return payload.dump();
}

OpenAIChatStream::OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                                   bool timings_per_token, bool return_progress)
    : identity_(std::move(identity)),
      include_usage_(include_usage),
      timings_per_token_(timings_per_token), return_progress_(return_progress) {}

std::string OpenAIChatStream::start() {
    if (started_ || finished_) { throw std::logic_error("OpenAI Chat stream already started"); }
    started_ = true;
    return chunk(identity_, Json{{"role", "assistant"}, {"content", ""}}, nullptr, include_usage_);
}

// Progress updates carry no output delta; like the terminal usage chunk, they use an empty delta
// object and keep the choice open (null finish_reason). The Engine publishes them before the
// first output delta, so the chunk always keeps the stream in its pre-content state.
std::string OpenAIChatStream::prompt_progress(const ninfer::PromptProgress& progress) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat prompt progress state");
    }
    if (!return_progress_) { throw std::logic_error("prompt progress was not requested"); }
    Json payload       = base_payload(identity_, "chat.completion.chunk");
    payload["choices"] = Json::array({stream_choice(Json::object(), nullptr)});
    if (include_usage_) { payload["usage"] = nullptr; }
    payload["prompt_progress"] = prompt_progress_json(progress);
    return event(std::move(payload));
}

void OpenAIChatStream::note_start(const ninfer::GenerationStart& start) {
    live_.prompt_tokens = start.prompt.prompt_tokens;
    live_.cached_tokens = start.reused_prompt_tokens;
    live_.begin         = std::chrono::steady_clock::now();
    live_.started       = true;
}

void OpenAIChatStream::note_live_delta(std::uint32_t committed_tokens) {
    if (!live_.first_delta_seen) {
        live_.first_delta_seen = true;
        live_.first_delta      = std::chrono::steady_clock::now();
    }
    live_.committed_tokens += committed_tokens;
}

nlohmann::json OpenAIChatStream::live_timings_json() const {
    if (!timings_per_token_ || !live_.started || !live_.first_delta_seen) { return nullptr; }
    const auto now        = std::chrono::steady_clock::now();
    const double prompt_ms =
        std::chrono::duration<double, std::milli>(live_.first_delta - live_.begin).count();
    const double predicted_ms =
        std::chrono::duration<double, std::milli>(now - live_.first_delta).count();
    return timings_json(make_completion_timings(
        live_.prompt_tokens, live_.cached_tokens,
        static_cast<std::uint32_t>(live_.committed_tokens), prompt_ms, predicted_ms, 0, 0));
}

std::string OpenAIChatStream::reasoning_delta(const std::string& text,
                                              std::uint32_t committed_tokens) {
    if (!started_ || finished_ || content_started_) {
        throw std::logic_error("invalid OpenAI Chat reasoning delta state");
    }
    reasoning_ += text;
    note_live_delta(committed_tokens);
    return chunk(identity_, Json{{"reasoning_content", text}}, nullptr, include_usage_,
                 live_timings_json());
}

std::string OpenAIChatStream::content_delta(const std::string& text,
                                            std::uint32_t committed_tokens) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat content delta state");
    }
    content_started_ = true;
    content_ += text;
    note_live_delta(committed_tokens);
    return chunk(identity_, Json{{"content", text}}, nullptr, include_usage_,
                 live_timings_json());
}

std::vector<std::string> OpenAIChatStream::finish(const GenerationOutcome& outcome) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat stream finish state");
    }
    finished_ = true;
    require_prefix(outcome.reasoning, reasoning_, "reasoning");
    require_prefix(outcome.text, content_, "content");

    // The last data chunk before [DONE] always carries the exact Engine completion timings: the
    // dedicated usage chunk when include_usage is on, otherwise the finish-reason chunk. With
    // timings_per_token the terminal chunks emitted here also carry the exact values.
    const Json final_timings       = timings_json(outcome_timings(outcome));
    const Json mid_terminal_timings = include_usage_ && !timings_per_token_ ? Json(nullptr)
                                                                             : final_timings;
    const Json intermediate_timings = timings_per_token_ ? final_timings : Json(nullptr);

    std::vector<std::string> events;
    const std::string reasoning_suffix = outcome.reasoning.substr(reasoning_.size());
    if (!reasoning_suffix.empty()) {
        if (content_started_) {
            throw std::logic_error("terminal reasoning appeared after streamed content");
        }
        events.push_back(chunk(identity_, Json{{"reasoning_content", reasoning_suffix}}, nullptr,
                               include_usage_, intermediate_timings));
    }
    const std::string content_suffix = outcome.text.substr(content_.size());
    if (!content_suffix.empty()) {
        events.push_back(chunk(identity_, Json{{"content", content_suffix}}, nullptr,
                               include_usage_, intermediate_timings));
    }

    if (!outcome.tool_calls.empty()) {
        const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
        events.push_back(chunk(identity_, Json{{"tool_calls", tool_calls_json(calls, true)}},
                               nullptr, include_usage_, intermediate_timings));
        events.push_back(chunk(identity_, Json::object(), "tool_calls", include_usage_,
                               mid_terminal_timings));
    } else {
        events.push_back(chunk(identity_, Json::object(), finish_reason(outcome.finish_reason),
                               include_usage_, mid_terminal_timings));
    }
    if (include_usage_) { events.push_back(usage_chunk(identity_, usage_from(outcome), final_timings)); }
    events.emplace_back("data: [DONE]\n\n");
    return events;
}

} // namespace ninfer::serve
