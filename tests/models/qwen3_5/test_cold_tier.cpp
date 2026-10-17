// SSD context tier: Native object-level mechanism on a real artifact.
//
// The exact reference for every cold copy is the Host replica it was made from: spilled bytes are
// read back from the segment file outside the store's read path and compared byte for byte, and
// restored Host replicas are compared with the bytes captured before the Host replica was
// released. Generation after a cold restore is compared with a Host-restore binding of the same
// checkpoint and with a full prefill that reuses nothing.

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen = models::qwen3_5;
using qwen::detail::ProgramImpl;

constexpr std::uint32_t kCapacity = 512;
constexpr std::uint32_t kChunk    = 128;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

std::vector<std::byte> read_file_range(const std::filesystem::path& path, std::uint64_t offset,
                                       std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    require(static_cast<bool>(file), "cold segment file could not be opened");
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> out(bytes);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    require(static_cast<std::uint64_t>(file.gcount()) == bytes, "cold segment file is short");
    return out;
}

void flip_file_byte(const std::filesystem::path& path, std::uint64_t offset) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    require(static_cast<bool>(file), "cold segment file could not be opened for corruption");
    file.seekg(static_cast<std::streamoff>(offset));
    char value = 0;
    file.read(&value, 1);
    value = static_cast<char>(value ^ 0x40);
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(&value, 1);
    file.flush();
    require(static_cast<bool>(file), "cold segment corruption was not written");
}

// One covered object of a checkpoint with its Host bytes.
struct ObjectBytes {
    bool state = false;
    qwen::detail::LogicalKVPageStore* pages = nullptr;
    qwen::detail::LogicalKVPageHandle page;
    qwen::detail::StateImageHandle image;
    std::vector<std::byte> bytes;
};

class ColdFixture {
public:
    ColdFixture(DeviceContext& device, ProgramImpl& program, qwen::Frontend& frontend,
                bool mtp)
        : device_(device), program_(program), frontend_(frontend), mtp_(mtp),
          empty_(program.physical_usage()) {
        require(program_.cold_store != nullptr, "cold fixture Program has no SSD tier");
    }

    std::vector<TokenId> tokens(std::uint32_t count, std::uint32_t salt = 0) const {
        std::vector<TokenId> ids(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            ids[i] = static_cast<TokenId>(1000 + (i * 7919U + salt * 104729U) % 30000U);
        }
        return ids;
    }

    qwen::RequestBasePlan request(std::vector<TokenId> ids, bool history,
                                  std::uint32_t output_tokens = 8) {
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = output_tokens;
        options.allow_prefix_reuse      = history;
        return program_.plan_request(
            qwen::PreparedPromptAccess::take(frontend_.prepare_tokens(std::move(ids))), options);
    }

    qwen::ContextProgress settle(bool cancelled = false) {
        std::atomic<bool> flag{cancelled};
        for (unsigned step = 0; step < 8; ++step) {
            device_.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device_.transfer_stream));
            auto progress = program_.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("context transaction did not settle");
    }

    std::vector<qwen::ColdCompletion> wait_cold() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        std::vector<qwen::ColdCompletion> out;
        while (std::chrono::steady_clock::now() < deadline) {
            auto completed = program_.poll_cold();
            out.insert(out.end(), completed.begin(), completed.end());
            if (program_.cold_jobs.empty()) { return out; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("cold job did not complete within 60 s");
    }

    qwen::ColdCompletion wait_one() {
        auto completed = wait_cold();
        require(completed.size() == 1, "expected exactly one cold completion");
        return completed.front();
    }

    qwen::SequenceHandle bind(const qwen::RequestBasePlan& base,
                              std::optional<qwen::CheckpointHandle> source) {
        auto candidate = program_.inspect_source(base, source);
        require(candidate.has_value(), "request has no admissible source");
        const auto reservation = program_.start_binding(base, {0}, *candidate, nullptr,
                                                        qwen::ExecutionUnitKind::Prefill, 0);
        require(static_cast<bool>(reservation), "binding was not reserved");
        auto progress = settle();
        require(progress.published && progress.sequence, "binding did not publish");
        return *progress.sequence;
    }

    // Runs the remaining prefill, then greedy decode; returns every sampled token.
    std::vector<TokenId> generate(qwen::SequenceHandle sequence, std::uint32_t decode_tokens,
                                  bool keep_checkpoint,
                                  std::optional<qwen::CheckpointHandle>* out) {
        std::vector<TokenId> sampled;
        for (unsigned iteration = 0; iteration < 64; ++iteration) {
            const std::array<qwen::ExecutionUnit, 1> units{
                {{sequence, qwen::ExecutionUnitKind::Prefill}}};
            require(static_cast<bool>(program_.reserve_units(units)), "prefill unit not reserved");
            auto step = program_.advance_prefill(sequence, nullptr, nullptr);
            if (step.capture_ready) { program_.skip_capture(sequence); }
            if (!step.complete) { continue; }
            require(step.pending && step.pending->tokens().size() == 1,
                    "prefill did not sample exactly one token");
            sampled.push_back(step.pending->tokens().front());
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = decode_tokens == 0}}};
            auto result = program_.commit(std::move(*step.pending), decision,
                                          runtime::CommitObservation::AllRows, nullptr);
            if (result.capture_ready[0]) { program_.skip_capture(sequence); }
            break;
        }
        require(sampled.size() == 1, "prefill did not complete");
        for (std::uint32_t index = 0; index < decode_tokens; ++index) {
            const std::array<qwen::ExecutionUnit, 1> units{
                {{sequence, qwen::ExecutionUnitKind::Decode, decode_tokens - index}}};
            require(static_cast<bool>(program_.reserve_units(units)), "decode unit not reserved");
            const std::array<qwen::SequenceHandle, 1> members{sequence};
            const std::array<runtime::RoundBudget, 1> budgets{
                {{.generated_tokens_remaining = decode_tokens - index}}};
            auto pending = program_.decode(members, budgets, nullptr, nullptr);
            require(!pending.tokens().empty(), "decode sampled nothing");
            // Speculative rounds may propose several tokens; accept only the first so every
            // route commits the same greedy sequence one token at a time.
            sampled.push_back(pending.tokens().front());
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = index + 1 == decode_tokens}}};
            auto result = program_.commit(std::move(pending), decision,
                                          runtime::CommitObservation::AllRows, nullptr);
            if (result.capture_ready[0]) { program_.skip_capture(sequence); }
        }
        auto finished = program_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed, "sequence was not consumed");
        if (out) {
            *out = finished.checkpoint;
        } else if (finished.checkpoint && !keep_checkpoint) {
            require(program_.release_checkpoint(*finished.checkpoint),
                    "generated checkpoint could not be released");
        }
        return sampled;
    }

    qwen::CheckpointHandle checkpoint(std::vector<TokenId> ids) {
        auto base           = request(std::move(ids), true);
        const auto sequence = bind(base, std::nullopt);
        std::optional<qwen::CheckpointHandle> made;
        (void)generate(sequence, 0, true, &made);
        require(made.has_value(), "seed request did not publish its checkpoint");
        return *made;
    }

    // Demotes every Device replica covered by the set to Host through the product path.
    void demote_all(std::span<const qwen::CheckpointHandle> set) {
        for (unsigned round = 0; round < 64; ++round) {
            const auto occupied = program_.physical_usage().occupied;
            const auto plan     = program_.plan_reclaim(
                set, {},
                {.state_slots      = occupied.state_slots,
                     .main_kv_pages    = occupied.main_kv_pages,
                     .backend_kv_pages = occupied.backend_kv_pages});
            if (plan.demotions.empty()) { return; }
            require(program_.start_demote(plan.demotions.front()), "demotion did not start");
            // A demotion whose Host copies already exist only drops Device replicas.
            if (program_.has_context_transaction()) {
                require(settle().published, "demotion did not publish");
            }
        }
        throw std::runtime_error("demotion did not converge");
    }

    std::vector<ObjectBytes> host_objects(qwen::CheckpointHandle handle) {
        std::vector<ObjectBytes> out;
        auto& p = program_;
        p.visit_checkpoint_objects(
            p.checkpoint(handle),
            [&](qwen::detail::StateImageHandle state) {
                if (!p.state_store->host_resident(state)) { return; }
                const auto view = p.state_store->host_view(state);
                ObjectBytes object{.state = true, .image = state};
                const auto bytes = p.state_images->host_layout().image_bytes;
                object.bytes.assign(view.data, view.data + bytes);
                out.push_back(std::move(object));
            },
            [&](qwen::detail::LogicalKVPageStore& pages, qwen::detail::LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                if (!pages.host_resident(page)) { return; }
                const auto replica = pages.host_replica(page);
                const auto view =
                    p.host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                const auto bytes = p.cold_store->object_bytes(
                    resource == runtime::ContextResourceClass::MainKV
                        ? ColdObjectClass::MainKVPage
                        : ColdObjectClass::BackendKVPage);
                ObjectBytes object{.pages = &pages, .page = page};
                object.bytes.assign(view.data(), view.data() + bytes);
                out.push_back(std::move(object));
            });
        return out;
    }

    const ColdSlot* cold_slot(const ObjectBytes& object) const {
        if (object.state) {
            const auto* replica = program_.state_store->cold_replica(object.image);
            return replica ? &replica->slot : nullptr;
        }
        const auto* replica = object.pages->cold_replica(object.page);
        return replica ? &replica->slot : nullptr;
    }

    std::uint64_t occupied_slots() const {
        const auto stats = program_.cold_store->stats();
        return stats.occupied_slots[0] + stats.occupied_slots[1] + stats.occupied_slots[2];
    }

    void expect_empty(std::string_view label) {
        require(!program_.has_context_transaction() && program_.cold_jobs.empty(), label);
        const auto now = program_.physical_usage();
        require(now.occupied == empty_.occupied && now.host_reserved_bytes == 0, label);
        require(occupied_slots() == 0 && program_.text_kv_pages->cold_only_pages() == 0 &&
                    program_.state_store->cold_only_states() == 0,
                std::string(label) + " (cold slots)");
    }

    // Spill, byte-compare file contents, release Host, restore, byte-compare Host contents.
    void roundtrip() {
        const auto source = checkpoint(tokens(200));
        const std::array set{source};
        demote_all(set);
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::Resident,
                "demoted checkpoint is not resident");
        const auto before = host_objects(source);
        const auto quote  = program_.cold_quote(set);
        const auto main_stride =
            program_.cold_store->object_bytes(ColdObjectClass::MainKVPage);
        // 200 tokens: three full Main pages and a partial tail page; the whole checkpoint,
        // tail included, is demoted.
        const std::uint32_t main_pages = (200 + 63) / 64;
        std::uint32_t backend_pages    = 0;
        if (program_.backend_kv_pages) {
            const auto frontier = mtp_ ? 199U : program_.checkpoint(source).backend_frontier;
            backend_pages       = (frontier + 63) / 64;
        }
        require(quote.write_objects == 1 + main_pages + backend_pages && !quote.lost &&
                    quote.device_only_objects == 0 && quote.host_bytes_retained == 0 &&
                    quote.whole && quote.block == qwen::ColdDemotionBlock::None,
                "spill quote does not cover the whole checkpoint");
        const auto spilled = program_.start_spill(set);
        std::cout << "  quote " << quote.write_objects << '/' << quote.write_bytes << " spill "
                  << static_cast<int>(spilled.status) << ' ' << spilled.objects << '/'
                  << spilled.bytes << '\n';
        require(spilled.status == qwen::ColdStartStatus::Started &&
                    spilled.objects == quote.write_objects && spilled.bytes == quote.write_bytes,
                "spill did not start with its quoted objects");
        require(program_.start_spill(set).status == qwen::ColdStartStatus::NothingToDo,
                "a second spill re-submitted objects pinned by the first");
        const auto spill = wait_one();
        require(spill.kind == qwen::ColdJobKind::Spill &&
                    spill.status == qwen::ColdJobStatus::Completed &&
                    spill.objects == quote.write_objects,
                "spill did not complete every object");
        std::uint32_t compared = 0;
        for (const auto& object : before) {
            const auto* slot = cold_slot(object);
            require(slot != nullptr, "a covered object has no SSD copy after spill");
            const auto file = read_file_range(program_.cold_store->segment_path(*slot),
                                              program_.cold_store->slot_offset(*slot),
                                              object.bytes.size());
            require(file == object.bytes, "SSD copy differs from its Host source bytes");
            ++compared;
        }
        require(compared == quote.write_objects, "not every spilled object was compared");
        const auto after_spill = program_.cold_quote(set);
        require(after_spill.write_objects == 0 && after_spill.cold_bytes == quote.write_bytes,
                "quote after spill does not show the current SSD copies");

        const auto host_before = program_.physical_usage().occupied.host_bytes;
        const auto released    = program_.release_cold_host(set);
        require(released == after_spill.host_bytes_released_after_spill &&
                    program_.physical_usage().occupied.host_bytes + released == host_before,
                "release_cold_host did not free exactly its quoted Host bytes");
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::ColdOnly,
                "released checkpoint is not cold-only");
        require(program_.text_kv_pages->cold_only_pages() == main_pages &&
                    program_.cold_quote(set).host_bytes_retained == 0 &&
                    program_.state_store->cold_only_states() == 1,
                "cold-only counters disagree with released objects");

        auto extension = request(tokens(260), true);
        const auto candidate = program_.inspect_source(extension, source);
        require(candidate && candidate->cold_restore, "cold-only source was not a legal candidate");
        std::uint64_t cold_bytes = 0;
        for (const auto& transfer : candidate->transfers) {
            if (transfer.direction == runtime::ContextTransferDirection::ColdToHost) {
                cold_bytes += transfer.work.payload_bytes;
            }
        }
        require(cold_bytes == after_spill.cold_bytes,
                "ColdToHost requirement differs from the cold-only bytes");
        const auto blocked = program_.start_binding(extension, {0}, *candidate, nullptr,
                                                    qwen::ExecutionUnitKind::Prefill, 0);
        require(!blocked.reserved && blocked.cold_restore && blocked.source_valid &&
                    !program_.has_context_transaction(),
                "binding a cold-only source did not ask for a restore");

        const auto restore = program_.start_restore(source);
        require(restore.status == qwen::ColdStartStatus::Started &&
                    restore.bytes == after_spill.cold_bytes && restore.host_bytes == restore.bytes,
                "restore did not start for the cold-only objects");
        require(program_.start_restore(source).job == restore.job,
                "a repeated restore did not join the running job");
        require(!program_.can_release_checkpoint(source),
                "a restoring checkpoint was releasable");
        const auto restored = wait_one();
        require(restored.kind == qwen::ColdJobKind::Restore &&
                    restored.status == qwen::ColdJobStatus::Completed && restored.lost.empty(),
                "restore failed");
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::Resident,
                "restored checkpoint is not resident");
        const auto after = host_objects(source);
        require(after.size() == before.size(), "restore changed the covered object set");
        for (std::size_t index = 0; index < before.size(); ++index) {
            require(after[index].bytes == before[index].bytes,
                    "restored Host bytes differ from the original Host replica");
        }
        // Redundant copies stay until dropped; drop releases every slot.
        require(occupied_slots() == quote.write_objects, "restore changed SSD occupancy");
        const auto dropped = program_.drop_cold(set);
        require(dropped == quote.write_bytes && occupied_slots() == 0,
                "drop_cold did not release every redundant SSD copy");
        require(program_.release_checkpoint(source), "restored checkpoint was not releasable");
        expect_empty("cold roundtrip leaked resources");
        std::cout << "  roundtrip: " << quote.write_objects << " objects, " << quote.write_bytes
                  << " bytes, spill " << spill.elapsed_ns / 1.0e6 << " ms, restore "
                  << restored.elapsed_ns / 1.0e6 << " ms\n";
        (void)main_stride;
    }

    // A shared prefix page is written once; a descendant spills only its own objects.
    void shared_pages() {
        const auto source = checkpoint(tokens(128));
        auto extension    = request(tokens(256), true);
        const auto bound  = bind(extension, source);
        std::optional<qwen::CheckpointHandle> made;
        (void)generate(bound, 0, true, &made);
        require(made.has_value(), "descendant was not retained");
        const auto descendant = *made;
        const std::array family{source, descendant};
        demote_all(family);
        const std::array one{source};
        const std::array other{descendant};
        const auto source_quote     = program_.cold_quote(one);
        const auto descendant_quote = program_.cold_quote(other);
        const auto family_quote     = program_.cold_quote(family);
        // Expected: distinct physical objects that are spillable across both checkpoints.
        std::set<std::pair<const void*, std::uint32_t>> distinct;
        for (const auto handle : family) {
            program_.visit_checkpoint_objects(
                program_.checkpoint(handle),
                [&](qwen::detail::StateImageHandle state) {
                    if (program_.state_store->can_spill(state)) {
                        distinct.emplace(program_.state_store.get(),
                                         program_.state_store->descriptor_index(state));
                    }
                },
                [&](qwen::detail::LogicalKVPageStore& pages, qwen::detail::LogicalKVPageHandle page,
                    runtime::ContextResourceClass) {
                    if (pages.can_spill(page)) {
                        distinct.emplace(&pages, pages.descriptor_index(page));
                    }
                });
        }
        std::cout << "  shared: source " << source_quote.write_objects << ", descendant "
                  << descendant_quote.write_objects << ", family " << family_quote.write_objects
                  << '\n';
        require(family_quote.write_objects == distinct.size() &&
                    family_quote.write_objects <
                        source_quote.write_objects + descendant_quote.write_objects,
                "family quote did not count each shared page once");
        const auto first = program_.start_spill(one);
        require(first.status == qwen::ColdStartStatus::Started, "source spill did not start");
        (void)wait_one();
        const auto remaining = program_.cold_quote(family);
        require(remaining.write_objects == family_quote.write_objects - source_quote.write_objects,
                "descendant quote did not exclude objects already on the SSD");
        const auto second = program_.start_spill(family);
        require(second.status == qwen::ColdStartStatus::Started &&
                    second.objects == remaining.write_objects,
                "family spill re-wrote shared objects");
        (void)wait_one();
        require(occupied_slots() == family_quote.write_objects,
                "shared objects occupy more than one slot");
        // Release Host for both, then delete the cold-only checkpoints without a restore.
        (void)program_.release_cold_host(family);
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::ColdOnly &&
                    program_.cold_residency(descendant) == ProgramImpl::ColdResidency::ColdOnly,
                "family did not become cold-only");
        require(program_.release_checkpoint(descendant) && program_.release_checkpoint(source),
                "cold-only checkpoints were not releasable");
        expect_empty("shared spill leaked SSD slots or descriptors");
    }

    // Spill in flight while the checkpoint is released: Host survives until the write retires.
    void release_during_spill() {
        const auto source = checkpoint(tokens(192));
        const std::array set{source};
        demote_all(set);
        const auto started = program_.start_spill(set);
        require(started.status == qwen::ColdStartStatus::Started, "spill did not start");
        require(program_.release_checkpoint(source), "spilling checkpoint was not releasable");
        (void)wait_cold();
        require(occupied_slots() == 0, "released objects kept their SSD copies");
        expect_empty("release during spill leaked Host or SSD resources");
    }

    // A corrupt copy is reported, never published, and leaves the checkpoint without content.
    void checksum_failure() {
        const auto source = checkpoint(tokens(200, 3));
        const std::array set{source};
        demote_all(set);
        require(program_.start_spill(set).status == qwen::ColdStartStatus::Started,
                "spill did not start");
        (void)wait_one();
        const auto objects = host_objects(source);
        (void)program_.release_cold_host(set);
        const ObjectBytes* target = nullptr;
        for (const auto& object : objects) {
            if (!object.state && cold_slot(object)) { target = &object; }
        }
        require(target != nullptr, "no spilled page to corrupt");
        const auto* slot = cold_slot(*target);
        flip_file_byte(program_.cold_store->segment_path(*slot),
                       program_.cold_store->slot_offset(*slot) + 4097);
        const auto host_before = program_.physical_usage().occupied.host_bytes;
        require(program_.start_restore(source).status == qwen::ColdStartStatus::Started,
                "restore did not start");
        const auto failed = wait_one();
        require(failed.status == qwen::ColdJobStatus::ChecksumFailed &&
                    failed.failed_objects == 1 && failed.lost.size() == 1 &&
                    failed.lost.front() == source,
                "checksum failure was not reported with its lost checkpoint");
        require(program_.physical_usage().occupied.host_bytes == host_before &&
                    program_.physical_usage().host_reserved_bytes == 0,
                "failed restore published or leaked its Host destination");
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::Lost,
                "corrupt checkpoint was not marked lost");
        auto extension = request(tokens(260, 3), true);
        require(!program_.inspect_source(extension, source).has_value(),
                "lost checkpoint remained a source candidate");
        require(program_.cold_store->stats().checksum_failures == 1,
                "checksum failure was not counted");
        require(program_.release_checkpoint(source), "lost checkpoint was not releasable");
        expect_empty("checksum failure leaked resources");
    }

    // Cancel restores nothing and keeps the cold-only content restorable.
    void cancel_restore() {
        const auto source = checkpoint(tokens(256, 5));
        const std::array set{source};
        demote_all(set);
        require(program_.start_spill(set).status == qwen::ColdStartStatus::Started,
                "spill did not start");
        (void)wait_one();
        (void)program_.release_cold_host(set);
        const auto restore = program_.start_restore(source);
        require(restore.status == qwen::ColdStartStatus::Started && restore.job &&
                    program_.cancel_cold(*restore.job),
                "restore could not be cancelled");
        const auto cancelled = wait_one();
        require(cancelled.status == qwen::ColdJobStatus::Cancelled ||
                    cancelled.status == qwen::ColdJobStatus::Completed,
                "cancelled restore reported a failure");
        if (cancelled.status == qwen::ColdJobStatus::Cancelled) {
            require(program_.cold_residency(source) == ProgramImpl::ColdResidency::ColdOnly &&
                        program_.physical_usage().host_reserved_bytes == 0,
                    "cancelled restore published or leaked Host");
            require(program_.start_restore(source).status == qwen::ColdStartStatus::Started,
                    "restore after cancel did not start");
            require(wait_one().status == qwen::ColdJobStatus::Completed,
                    "restore after cancel failed");
        }
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::Resident,
                "restore after cancel is not resident");
        (void)program_.drop_cold(set);
        require(program_.release_checkpoint(source), "restored checkpoint was not releasable");
        expect_empty("cancelled restore leaked resources");
    }

    // Greedy tokens after a cold restore equal Host restore and full prefill.
    void generation_equality() {
        const auto prompt = tokens(300, 11);
        const auto prefix = std::vector<TokenId>(prompt.begin(), prompt.begin() + 200);
        constexpr std::uint32_t kDecode = 16;

        auto fresh_base   = request(prompt, false, kDecode + 1);
        const auto fresh  = generate(bind(fresh_base, std::nullopt), kDecode, false, nullptr);

        const auto source = checkpoint(prefix);
        const std::array set{source};
        demote_all(set);
        auto host_base   = request(prompt, true, kDecode + 1);
        const auto host  = generate(bind(host_base, source), kDecode, false, nullptr);

        demote_all(set);
        require(program_.start_spill(set).status == qwen::ColdStartStatus::Started,
                "spill did not start");
        (void)wait_one();
        (void)program_.release_cold_host(set);
        require(program_.cold_residency(source) == ProgramImpl::ColdResidency::ColdOnly,
                "generation source is not cold-only");
        require(program_.start_restore(source).status == qwen::ColdStartStatus::Started,
                "restore did not start");
        require(wait_one().status == qwen::ColdJobStatus::Completed, "restore failed");
        auto cold_base  = request(prompt, true, kDecode + 1);
        const auto cold = generate(bind(cold_base, source), kDecode, false, nullptr);

        const auto print = [](std::string_view label, const std::vector<TokenId>& ids) {
            std::cout << "  " << label << ":";
            for (const auto id : ids) { std::cout << ' ' << id; }
            std::cout << '\n';
        };
        print("full prefill", fresh);
        print("host restore", host);
        print("cold restore", cold);
        require(cold == host, "cold-restore generation differs from Host-restore generation");
        require(cold == fresh, "cold-restore generation differs from full prefill");
        (void)program_.drop_cold(set);
        require(program_.release_checkpoint(source), "generation source was not releasable");
        expect_empty("generation equality leaked resources");
    }

    // Destroying the Program with jobs in flight joins I/O before Host memory is released.
    void destroy_with_jobs() {
        const auto source = checkpoint(tokens(256, 7));
        const std::array set{source};
        demote_all(set);
        require(program_.start_spill(set).status == qwen::ColdStartStatus::Started,
                "spill did not start");
    }

private:
    DeviceContext& device_;
    ProgramImpl& program_;
    qwen::Frontend& frontend_;
    bool mtp_ = false;
    qwen::PhysicalUsageSnapshot empty_;
};

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        require(argc <= 3, "usage: ninfer_qwen3_5_cold_tier_test [none|mtp] [cold-dir]");
        const std::string backend_name = argc >= 2 ? argv[1] : "none";
        require(backend_name == "none" || backend_name == "mtp", "backend must be none or mtp");
        const auto backend =
            backend_name == "mtp" ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
        std::filesystem::path directory;
        if (argc == 3) {
            directory = argv[2];
        } else if (const char* env = std::getenv("NINFER_TEST_COLD_DIR"); env && *env) {
            directory = env;
        } else {
            directory = std::filesystem::current_path() / "cold-tier-test";
        }
        std::filesystem::create_directories(directory);

        DeviceContext device;
        models::LoadOptions selected;
        selected.speculative = backend;
        auto model           = qwen::load_model(artifact, selected, device);
        qwen::execution::Parameters parameters(*model);
        auto frontend = qwen::make_frontend(model->resources(),
                                            {.vision_enabled = false, .max_context = kCapacity});
        EngineOptions options;
        options.max_context              = kCapacity;
        options.kv_capacity              = KvCapacityPolicy::explicit_capacity(2 * kCapacity);
        options.prefill_chunk            = kChunk;
        options.max_concurrency          = 2;
        options.kv_cache                 = KvCacheStorage::Fp8E4M3Row256;
        options.use_cuda_graph           = false;
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0U : 3U;
        options.context_cache.device_state_slots  = 2;
        options.context_cache.host_capacity_bytes = 512ULL * 1024 * 1024;
        options.context_cache.cold_directory      = directory;
        options.context_cache.cold_capacity_bytes = 2ULL * 1024 * 1024 * 1024;
        auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
        auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 16);
        std::uint64_t written = 0, read = 0;
        {
            ProgramImpl program(parameters, *plan, device, {});
            const auto probe = program.cold_store->probe();
            std::cout << "  probe: write " << probe.write_bytes_per_second() / 1.0e9
                      << " GB/s, read " << probe.read_bytes_per_second() / 1.0e9 << " GB/s\n";
            ColdFixture fixture(device, program, frontend, backend == SpeculativeBackend::Mtp);
            fixture.roundtrip();
            fixture.shared_pages();
            fixture.release_during_spill();
            fixture.checksum_failure();
            fixture.cancel_restore();
            if (backend == SpeculativeBackend::None) { fixture.generation_equality(); }
            fixture.destroy_with_jobs();
            const auto stats = program.cold_store->stats();
            written          = stats.write_bytes;
            read             = stats.read_bytes;
        }
        std::cout << "  store write bytes " << written << ", read bytes " << read << '\n';
        std::cout << "OK cold tier " << backend_name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL cold tier: " << error.what() << '\n';
        return 1;
    }
}
