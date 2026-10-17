// SSD context tier policy through the public Engine: SSD capacity eviction, the demote window's
// age rule and the watermark pre-write.
//
// A round-robin driver advances every history by one turn per round. Each turn restores its
// older checkpoint (from the SSD when it left Host), so the restores keep creating Host pressure
// that an admission alone does not, and each new turn writes new content, which fills the SSD.
// The exact reference for every generated token sequence is an Engine without the tier and
// without prefix reuse, which computes the same prompt by full prefill. Whatever route a turn
// takes (restore, erased and recomputed, evicted and recomputed), its tokens must equal it.

#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kPromptTokens = 256;
constexpr std::uint32_t kOutputTokens = 8;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

struct Policy {
    std::uint32_t window_seconds = 600;
    std::uint32_t watermark      = 0;
    std::uint64_t cold_capacity  = 8ULL << 30U;
};

ninfer::EngineOptions options(const char* artifact, const std::filesystem::path& cold,
                              const Policy& policy) {
    ninfer::EngineOptions out;
    out.artifact_path                    = artifact;
    out.max_context                      = 1024;
    out.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    out.prefill_chunk                    = kPromptTokens;
    out.max_concurrency                  = 1;
    out.max_pending_requests             = 2;
    out.use_cuda_graph                   = false;
    out.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    out.context_cache.device_state_slots = 1;
    // About two StateImages: every further history must leave Host for the SSD or be erased.
    out.context_cache.host_capacity_bytes = 2ULL * 153954304 + (64ULL << 20U);
    if (!cold.empty()) {
        out.context_cache.cold_directory                 = cold;
        out.context_cache.cold_capacity_bytes            = policy.cold_capacity;
        out.context_cache.cold_demote_window_seconds     = policy.window_seconds;
        out.context_cache.cold_prewrite_watermark_percent = policy.watermark;
    }
    return out;
}

std::vector<ninfer::TokenId> prompt(std::uint32_t family) {
    std::vector<ninfer::TokenId> out(kPromptTokens);
    for (std::uint32_t index = 0; index < kPromptTokens; ++index) {
        out[index] = static_cast<ninfer::TokenId>(1000 + (family * 7919 + index * 131) % 20000);
    }
    return out;
}

std::vector<ninfer::TokenId> continuation(std::uint32_t family) {
    auto out = prompt(family);
    for (std::uint32_t index = 0; index < 32; ++index) {
        out.push_back(static_cast<ninfer::TokenId>(2000 + (family * 31 + index * 17) % 9000));
    }
    return out;
}

ninfer::RequestOptions request(bool reuse) {
    ninfer::RequestOptions out;
    out.execution.requested_output_tokens = kOutputTokens;
    out.execution.sampling.temperature    = 0.0F;
    out.execution.allow_prefix_reuse      = reuse;
    out.stop.include_model_defaults       = false;
    return out;
}

void generate(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens) {
    (void)engine.generate(engine.prepare_tokens(tokens), request(true));
}

void wait_idle(ninfer::Engine& engine) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto stats = engine.runtime_stats();
        if (stats.cold_jobs == 0 && stats.host_context_reserved_bytes == 0) { return; }
        require(std::chrono::steady_clock::now() < deadline,
                "SSD jobs or Host reservations were left behind");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void print_cold(std::string_view label, const ninfer::RuntimeStats& stats) {
    std::cout << "  " << label << ": spills " << stats.cold_spills_started << '/'
              << stats.cold_spills_completed << " erased " << stats.cold_erased_checkpoints
              << " expired " << stats.cold_expired_erasures << " evictions "
              << stats.cold_evictions << " (" << stats.cold_evicted_bytes << " B) prewrites "
              << stats.cold_prewrites << " wasted " << stats.cold_prewrite_wasted
              << " restores " << stats.cold_restores_completed << " cold-only states "
              << stats.cold_only_states << " write " << stats.cold_write_bytes << '\n';
}

// Greedy tokens of a continuation by full prefill without the tier and without reuse.
std::vector<ninfer::TokenId> reference(const char* artifact, std::uint32_t family) {
    ninfer::Engine baseline(options(artifact, {}, {}));
    const auto result =
        baseline.generate(baseline.prepare_tokens(continuation(family)), request(false));
    require(result.reused_prompt_tokens == 0, "the reference reused history");
    return result.generated_token_ids;
}

std::filesystem::path fresh_directory(const std::filesystem::path& root, std::string_view name) {
    const auto directory = root / name;
    std::filesystem::create_directories(directory);
    return directory;
}

// Turn `round` of history `family`: the history's prompt extended by one prefill chunk per round,
// so every turn reuses the (chunk-aligned) input checkpoint its previous turn left behind and the
// reference's full prefill has the same chunk geometry.
std::vector<ninfer::TokenId> turn(std::uint32_t family, std::uint32_t round) {
    auto out = prompt(family);
    for (std::uint32_t index = 0; index < kPromptTokens * round; ++index) {
        out.push_back(static_cast<ninfer::TokenId>(2000 + (family * 31 + index * 17) % 9000));
    }
    return out;
}

// Runs `rounds` rounds over `families` histories on an Engine with `policy`; every generated
// sequence must equal the full-prefill reference. Returns the final statistics.
ninfer::RuntimeStats drive(const char* artifact, const std::filesystem::path& root,
                           std::string_view name, const Policy& policy, std::uint32_t families,
                           std::uint32_t rounds, std::chrono::milliseconds pause) {
    std::vector<std::vector<ninfer::TokenId>> expected;
    {
        ninfer::Engine baseline(options(artifact, {}, {}));
        for (std::uint32_t round = 0; round < rounds; ++round) {
            for (std::uint32_t family = 1; family <= families; ++family) {
                const auto result = baseline.generate(
                    baseline.prepare_tokens(turn(family, round)), request(false));
                require(result.reused_prompt_tokens == 0, "the reference reused history");
                expected.push_back(result.generated_token_ids);
            }
        }
    }
    ninfer::Engine engine(options(artifact, fresh_directory(root, name), policy));
    std::size_t index = 0;
    for (std::uint32_t round = 0; round < rounds; ++round) {
        for (std::uint32_t family = 1; family <= families; ++family, ++index) {
            const auto result =
                engine.generate(engine.prepare_tokens(turn(family, round)), request(true));
            require(result.generated_token_ids == expected[index],
                    "a turn changed the greedy tokens (round " + std::to_string(round) +
                        ", history " + std::to_string(family) + ")");
        }
        wait_idle(engine);
        print_cold("round " + std::to_string(round), engine.runtime_stats());
        if (pause.count()) { std::this_thread::sleep_for(pause); }
    }
    wait_idle(engine);
    return engine.runtime_stats();
}

// Histories inside the window are written; when the SSD is full its oldest content is evicted.
void full_ssd_evicts_the_oldest(const char* artifact, const std::filesystem::path& root) {
    const auto stats = drive(artifact, root, "policy-evict",
                             {.window_seconds = 600, .watermark = 0,
                              .cold_capacity = 700ULL << 20U},
                             12, 3, std::chrono::milliseconds(0));
    print_cold("eviction", stats);
    require(stats.cold_spills_completed >= 1, "nothing was written to the SSD");
    require(stats.cold_restores_completed >= 1, "no history was restored from the SSD");
    require(stats.cold_evictions >= 1, "a full SSD did not evict its oldest content");
    require(stats.cold_evicted_bytes != 0, "an eviction freed no bytes");
}

// Content idle past the window is erased on demand, never written.
void expired_content_is_erased(const char* artifact, const std::filesystem::path& root) {
    const auto stats = drive(artifact, root, "policy-age",
                             {.window_seconds = 2, .watermark = 0}, 6, 3,
                             std::chrono::milliseconds(2600));
    print_cold("age rule", stats);
    require(stats.cold_expired_erasures >= 1,
            "no content past the demote window was erased on demand");
}

// Below the Host watermark the next victims are written ahead without releasing Host; a
// later pressure demotion only releases them and the history restores with the same tokens.
void watermark_prewrites(const char* artifact, const std::filesystem::path& root,
                         const std::vector<ninfer::TokenId>& expected) {
    ninfer::Engine engine(options(artifact, fresh_directory(root, "policy-prewrite"),
                                  {.window_seconds = 600, .watermark = 90}));
    // The pre-write runs from the worker loop, which only turns while requests are served.
    std::uint32_t family = 1;
    for (; family <= 12 && engine.runtime_stats().cold_prewrites == 0; ++family) {
        generate(engine, prompt(family));
    }
    wait_idle(engine);
    print_cold("pre-write", engine.runtime_stats());
    require(engine.runtime_stats().cold_prewrites >= 1,
            "no checkpoint was pre-written below the Host watermark");
    for (; family <= 24 && engine.runtime_stats().cold_only_states < 1; ++family) {
        generate(engine, prompt(family));
    }
    wait_idle(engine);
    print_cold("after pressure", engine.runtime_stats());
    const auto result =
        engine.generate(engine.prepare_tokens(continuation(1)), request(true));
    require(result.generated_token_ids == expected,
            "a pre-written history changed the greedy tokens");
    wait_idle(engine);
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        std::filesystem::path root;
        if (argc >= 2) {
            root = argv[1];
        } else if (const char* env = std::getenv("NINFER_TEST_COLD_DIR"); env && *env) {
            root = env;
        } else {
            root = std::filesystem::current_path() / "cold-policy-test";
        }
        std::filesystem::create_directories(root);
        const auto expected = reference(artifact, 1);
        full_ssd_evicts_the_oldest(artifact, root);
        expired_content_is_erased(artifact, root);
        watermark_prewrites(artifact, root, expected);
        std::cout << "OK cold policy\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL cold policy: " << error.what() << '\n';
        return 1;
    }
}
