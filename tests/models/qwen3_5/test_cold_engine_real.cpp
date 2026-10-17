// SSD context tier through the public Engine.
//
// Host pressure must write the oldest history to the SSD instead of erasing it, and a later
// request must restore it SSD -> Host -> Device. The exact reference for every generated token
// sequence is a separate Engine without the tier and without prefix reuse, which computes the
// same prompt by full prefill in the same chunk geometry. A corrupted SSD copy must fall back to
// prefill with the same tokens, and a request cancelled while its restore runs must leave no
// job or Host reservation behind.

#include "ninfer/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

ninfer::EngineOptions options(const char* artifact, const std::filesystem::path& cold) {
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
    out.context_cache.host_capacity_bytes = 2ULL * 153954304 + (64ULL << 20);
    if (!cold.empty()) {
        out.context_cache.cold_directory      = cold;
        out.context_cache.cold_capacity_bytes = 8ULL << 30;
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

std::string tokens(const std::vector<ninfer::TokenId>& values) {
    std::string out;
    for (const auto value : values) { out += ' ' + std::to_string(value); }
    return out;
}

void print_cold(std::string_view label, const ninfer::RuntimeStats& stats) {
    std::cout << "  " << label << ": spills " << stats.cold_spills_started << '/'
              << stats.cold_spills_completed << " failed " << stats.cold_spills_failed
              << " erased " << stats.cold_erased_checkpoints << " restores "
              << stats.cold_restores_started << '/' << stats.cold_restores_completed
              << " cancelled " << stats.cold_restores_cancelled << " failed "
              << stats.cold_restores_failed << " cold-only states " << stats.cold_only_states
              << " pages " << stats.cold_only_main_kv_pages << " write " << stats.cold_write_bytes
              << " (24h " << stats.cold_write_bytes_24h << ") read " << stats.cold_read_bytes
              << " checksum failures " << stats.cold_checksum_failures << " reclaim-erased "
              << stats.reclaim_erased_checkpoints << '\n';
}

// Fillers are newer histories; reclaim takes the oldest unreused ones first.
void push_to_ssd(ninfer::Engine& engine, std::uint32_t& next_filler, std::uint32_t states) {
    for (std::uint32_t attempt = 0; attempt < 12; ++attempt) {
        const auto before = engine.runtime_stats();
        if (before.cold_only_states >= states) { return; }
        (void)engine.generate(engine.prepare_tokens(prompt(100 + next_filler++)), request(true));
    }
    require(engine.runtime_stats().cold_only_states >= states,
            "Host pressure did not move the oldest histories to the SSD");
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

void corrupt_segments(const std::filesystem::path& directory) {
    std::uint32_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("seg_")) { continue; }
        std::fstream file(entry.path(), std::ios::binary | std::ios::in | std::ios::out);
        require(static_cast<bool>(file), "segment file could not be opened");
        const auto size = std::filesystem::file_size(entry.path());
        for (std::uint64_t offset = 4096; offset < size; offset += 1ULL << 20U) {
            file.seekg(static_cast<std::streamoff>(offset));
            char value = 0;
            file.read(&value, 1);
            value = static_cast<char>(value ^ 0x5A);
            file.seekp(static_cast<std::streamoff>(offset));
            file.write(&value, 1);
        }
        file.flush();
        ++files;
    }
    require(files != 0, "no SSD segment file to corrupt");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        std::filesystem::path directory;
        if (argc >= 2) {
            directory = argv[1];
        } else if (const char* env = std::getenv("NINFER_TEST_COLD_DIR"); env && *env) {
            directory = env;
        } else {
            directory = std::filesystem::current_path() / "cold-engine-test";
        }
        std::filesystem::create_directories(directory);

        // Family 1 is restored (and later corrupted), family 2 is cancelled during its restore.
        std::vector<std::vector<ninfer::TokenId>> reference(4);
        {
            ninfer::Engine baseline(options(artifact, {}));
            for (std::uint32_t family = 1; family <= 2; ++family) {
                const auto result = baseline.generate(
                    baseline.prepare_tokens(continuation(family)), request(false));
                require(result.reused_prompt_tokens == 0, "the reference reused history");
                reference[family] = result.generated_token_ids;
                std::cout << "  reference " << family << ':' << tokens(reference[family]) << '\n';
            }
        }
        // Families 1 and 2 are the oldest histories, so Host pressure moves them to the SSD.
        const auto seeded = [&](std::unique_ptr<ninfer::Engine>& engine) {
            engine = std::make_unique<ninfer::Engine>(options(artifact, directory));
            for (std::uint32_t family = 1; family <= 3; ++family) {
                (void)engine->generate(engine->prepare_tokens(prompt(family)), request(true));
            }
            require(engine->runtime_stats().cold_enabled, "the SSD tier is not enabled");
            std::uint32_t filler = 0;
            push_to_ssd(*engine, filler, 2);
            print_cold("after pressure", engine->runtime_stats());
            require(engine->runtime_stats().cold_spills_completed >= 2, "no spill completed");
        };

        std::unique_ptr<ninfer::Engine> engine;
        seeded(engine);
        // 1. SSD -> Host -> Device restore of the oldest history.
        const auto restored =
            engine->generate(engine->prepare_tokens(continuation(1)), request(true));
        std::cout << "  cold restore:" << tokens(restored.generated_token_ids) << '\n';
        require(restored.scheduling.cold_restores == 1 && restored.scheduling.cold_to_host_bytes &&
                    restored.reused_prompt_tokens == kPromptTokens,
                "the request did not reuse its history through an SSD restore");
        require(restored.generated_token_ids == reference[1],
                "SSD restore changed the greedy tokens");
        std::cout << "  restore wait " << restored.scheduling.cold_restore_wait_ns / 1000
                  << " us, " << restored.scheduling.cold_to_host_bytes << " bytes\n";
        wait_idle(*engine);

        // 2. Cancel as soon as the restore of family 2 has started.
        const auto before_cancel = engine->runtime_stats();
        std::atomic<bool> cancel{false};
        std::thread watcher([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!cancel.load() && std::chrono::steady_clock::now() < deadline) {
                if (engine->runtime_stats().cold_restores_started >
                    before_cancel.cold_restores_started) {
                    cancel = true;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            cancel = true;
        });
        const auto cancelled = engine->generate(
            engine->prepare_tokens(continuation(2)), request(true), nullptr,
            ninfer::CancellationView([&] { return cancel.load(); }));
        watcher.join();
        wait_idle(*engine);
        const auto after_cancel = engine->runtime_stats();
        const bool cancelled_restore =
            after_cancel.cold_restores_cancelled > before_cancel.cold_restores_cancelled;
        std::cout << "  cancellation during restore: "
                  << (cancelled_restore ? "observed" : "restore completed first")
                  << ", finish " << static_cast<int>(cancelled.finish_reason) << '\n';
        const auto retried =
            engine->generate(engine->prepare_tokens(continuation(2)), request(true));
        require(retried.generated_token_ids == reference[2],
                "generation after a cancelled restore changed the greedy tokens");
        require(!cancelled_restore || retried.scheduling.cold_restores == 1,
                "a cancelled restore did not leave its SSD copy usable");
        wait_idle(*engine);
        const auto stats = engine->runtime_stats();
        print_cold("after cancellation", stats);
        require(stats.cold_write_bytes_24h == stats.cold_write_bytes,
                "the 24 h SSD write window does not reconcile with the lifetime total");
        engine.reset();

        // 3. A fresh process-volatile tier; corrupt every SSD copy before restoring family 1.
        seeded(engine);
        corrupt_segments(directory);
        const auto before_corrupt = engine->runtime_stats();
        const auto fallback =
            engine->generate(engine->prepare_tokens(continuation(1)), request(true));
        const auto after_corrupt = engine->runtime_stats();
        std::cout << "  checksum fallback:" << tokens(fallback.generated_token_ids)
                  << " reason " << static_cast<int>(fallback.admission.fallback_reason)
                  << " reused " << fallback.reused_prompt_tokens << '\n';
        print_cold("after corruption", after_corrupt);
        require(after_corrupt.cold_checksum_failures > before_corrupt.cold_checksum_failures &&
                    after_corrupt.cold_restores_failed > before_corrupt.cold_restores_failed,
                "the corrupted SSD copy was not detected");
        require(!after_corrupt.cold_disabled &&
                    after_corrupt.cold_erased_checkpoints == before_corrupt.cold_erased_checkpoints,
                "a corrupted copy disabled the tier or Host victims were erased");
        require(fallback.admission.fallback_reason == ninfer::AdmissionFallbackReason::ColdChecksum,
                "the checksum fallback reason was not reported");
        require(fallback.scheduling.cold_restores == 0 && fallback.reused_prompt_tokens == 0,
                "a corrupted SSD copy was used");
        require(fallback.generated_token_ids == reference[1],
                "prefill after a checksum failure changed the greedy tokens");
        wait_idle(*engine);
        std::cout << "OK cold engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL cold engine: " << error.what() << '\n';
        return 1;
    }
}
