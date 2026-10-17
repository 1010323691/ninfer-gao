#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"

#include <algorithm>
#include <array>
#include <exception>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// SSD tier mechanism. Spill copies immutable Host replicas to cold slots; restore reads cold-only
// objects of one checkpoint into a complete hidden Host destination and publishes it at once.
// Neither touches the GPU, so neither is a context transaction. Which checkpoints move and when
// is Runtime policy.

namespace ninfer::models::qwen3_5::detail {
namespace {

ColdObjectClass object_class(runtime::ContextResourceClass resource) {
    switch (resource) {
    case runtime::ContextResourceClass::MainKV:
        return ColdObjectClass::MainKVPage;
    case runtime::ContextResourceClass::BackendKV:
        return ColdObjectClass::BackendKVPage;
    case runtime::ContextResourceClass::State:
        return ColdObjectClass::StateImage;
    }
    throw std::logic_error("unknown context resource class");
}

// Deduplicates physical objects shared by several checkpoints or histories.
struct ObjectSet {
    std::unordered_set<std::uint32_t> states, main, backend;

    bool insert_state(const StateImageStore& store, StateImageHandle state) {
        return states.insert(store.descriptor_index(state)).second;
    }

    bool insert_page(const LogicalKVPageStore& store, LogicalKVPageHandle page,
                     runtime::ContextResourceClass resource) {
        auto& seen = resource == runtime::ContextResourceClass::MainKV ? main : backend;
        return seen.insert(store.descriptor_index(page)).second;
    }
};

// Identifies one physical KV page across the Main and backend stores.
std::uint64_t shared_page_key(const LogicalKVPageStore& store, LogicalKVPageHandle page,
                              runtime::ContextResourceClass resource) {
    return (static_cast<std::uint64_t>(resource == runtime::ContextResourceClass::MainKV) << 32U) |
           store.descriptor_index(page);
}

// Identifies one StateImage in the same key space as shared_page_key (bit 33 marks a state).
std::uint64_t shared_state_key(const StateImageStore& store, StateImageHandle state) {
    return (1ULL << 33U) | store.descriptor_index(state);
}

} // namespace

ProgramImpl::ColdResidency ProgramImpl::cold_residency(CheckpointHandle handle) const {
    if (!valid_checkpoint(handle)) { return ColdResidency::Lost; }
    bool lost = false, cold = false;
    visit_checkpoint_objects(
        checkpoint(handle),
        [&](StateImageHandle state) {
            lost |= !state_store->has_replica(state);
            cold |= state_store->cold_only(state);
        },
        [&](const LogicalKVPageStore& pages, LogicalKVPageHandle page,
            runtime::ContextResourceClass) {
            lost |= !pages.has_replica(page);
            cold |= pages.cold_only(page);
        });
    return lost ? ColdResidency::Lost : cold ? ColdResidency::ColdOnly : ColdResidency::Resident;
}

std::optional<ColdJobId> ProgramImpl::cold_restore_job(CheckpointHandle handle) const {
    for (const auto& [id, job] : cold_jobs) {
        if (job.kind == ColdJobKind::Restore &&
            std::find(job.checkpoints.begin(), job.checkpoints.end(), handle) !=
                job.checkpoints.end()) {
            return id;
        }
    }
    return std::nullopt;
}

ColdUsage ProgramImpl::cold_usage() const {
    ColdUsage out;
    if (!cold_store) { return out; }
    const auto stats       = cold_store->stats();
    out.enabled            = true;
    out.disabled           = stats.disabled;
    out.capacity_bytes     = stats.capacity_bytes;
    out.segment_bytes      = stats.segment_bytes;
    out.occupied_bytes     = stats.occupied_bytes;
    out.segments           = stats.segments;
    out.main_pages         = stats.occupied_slots[0];
    out.backend_pages      = stats.occupied_slots[1];
    out.states             = stats.occupied_slots[2];
    out.cold_only_main_pages    = text_kv_pages->cold_only_pages();
    out.cold_only_backend_pages = backend_kv_pages ? backend_kv_pages->cold_only_pages() : 0;
    out.cold_only_states        = state_store->cold_only_states();
    out.write_count          = stats.write_count;
    out.read_count           = stats.read_count;
    out.write_bytes          = stats.write_bytes;
    out.read_bytes           = stats.read_bytes;
    out.write_bytes_24h      = stats.write_bytes_24h;
    out.write_failures       = stats.write_failures;
    out.read_failures        = stats.read_failures;
    out.checksum_failures    = stats.checksum_failures;
    out.inflight_write_bytes = stats.inflight_write_bytes;
    out.inflight_read_bytes  = stats.inflight_read_bytes;
    out.jobs                 = static_cast<std::uint32_t>(cold_jobs.size());
    out.probe_write_bytes_per_second = cold_store->probe().write_bytes_per_second();
    out.probe_read_bytes_per_second  = cold_store->probe().read_bytes_per_second();
    return out;
}

std::unordered_map<std::uint64_t, std::uint32_t>
ProgramImpl::victim_page_holders(std::span<const CheckpointHandle> handles) const {
    std::unordered_map<std::uint64_t, std::uint32_t> holders;
    for (const auto handle : handles) {
        if (!valid_checkpoint(handle)) { continue; }
        ObjectSet once;
        visit_checkpoint_objects(
            checkpoint(handle),
            [&](StateImageHandle state) {
                if (!once.insert_state(*state_store, state)) { return; }
                ++holders[shared_state_key(*state_store, state)];
            },
            [&](const LogicalKVPageStore& pages, LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                if (!once.insert_page(pages, page, resource)) { return; }
                ++holders[shared_page_key(pages, page, resource)];
            });
    }
    return holders;
}

ColdQuote ProgramImpl::cold_quote(std::span<const CheckpointHandle> handles) const {
    ColdQuote out;
    out.whole = true;
    out.block = ColdDemotionBlock::None;
    const auto block = [&](ColdDemotionBlock reason) {
        if (out.whole) {
            out.whole = false;
            out.block = reason;
        }
    };
    if (!cold_store || cold_store->disabled()) { block(ColdDemotionBlock::Unavailable); }
    if (handles.empty()) { block(ColdDemotionBlock::Lost); }
    ObjectSet seen;
    std::array<std::uint64_t, kColdObjectClassCount> writes{};
    // Objects that become cold-only (no Device replica) when their Host replica is released.
    std::uint64_t new_cold_states = 0, new_cold_main = 0, new_cold_backend = 0;
    const auto image = state_images->host_layout().image_bytes;
    auto victim_holders = victim_page_holders(handles);
    for (const auto handle : handles) {
        if (!valid_checkpoint(handle)) {
            block(ColdDemotionBlock::Lost);
            continue;
        }
        // A Device-only object stays where it is; whole demotion covers the objects that leave Host.
        // Whole-demotion facts of this checkpoint; shared objects are judged per checkpoint and
        // their bytes count once.
        auto local             = ColdDemotionBlock::None;
        std::size_t releasable = 0;
        std::size_t host_bytes = 0;
        const auto note        = [&](ColdDemotionBlock reason) {
            if (local == ColdDemotionBlock::None) { local = reason; }
        };
        visit_checkpoint_objects(
            checkpoint(handle),
            [&](StateImageHandle state) {
                const bool first = seen.insert_state(*state_store, state);
                if (!state_store->has_replica(state)) {
                    out.lost = true;
                    note(ColdDemotionBlock::Lost);
                } else if (state_store->cold_only(state)) {
                    if (first) {
                        out.restore_bytes += image;
                        out.restore_host_bytes += image;
                        ++out.restore_objects;
                    }
                } else if (state_store->host_resident(state)) {
                    const bool current = state_store->cold_replica_current(state);
                    const bool spill   = !current && state_store->can_spill(state);
                    if (first && current) { out.cold_bytes += image; }
                    if (first && spill) {
                        out.write_bytes += image;
                        ++out.write_objects;
                        ++writes[static_cast<std::size_t>(ColdObjectClass::StateImage)];
                    }
                    if (first) { host_bytes += image; }
                    if (!current && !spill) {
                        note(state_store->cold_busy(state) ? ColdDemotionBlock::Busy
                                                           : ColdDemotionBlock::Unspillable);
                    } else if (!state_store->host_releasable_after_spill(state)) {
                        note(state_store->cold_busy(state) ? ColdDemotionBlock::Busy
                                                           : ColdDemotionBlock::Unreleasable);
                    } else if (state_store->checkpoint_references(state) >
                               victim_holders[shared_state_key(*state_store, state)]) {
                        // Another checkpoint holds this image: it stays in Host with its copy.
                    } else if (first) {
                        releasable += image;
                        if (!state_store->device_resident(state)) { ++new_cold_states; }
                    }
                } else if (state_store->cold_replica_current(state)) {
                    if (first) { out.cold_bytes += image; }
                } else {
                    if (first) { ++out.device_only_objects; }
                }
            },
            [&](const LogicalKVPageStore& pages, LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                const bool first  = seen.insert_page(pages, page, resource);
                const bool main   = &pages == text_kv_pages.get();
                const auto stride = main ? text_host_kv_page_stride : backend_host_kv_page_stride;
                if (!pages.has_replica(page)) {
                    out.lost = true;
                    note(ColdDemotionBlock::Lost);
                } else if (pages.cold_only(page)) {
                    if (first) {
                        out.restore_bytes += stride;
                        out.restore_host_bytes += stride;
                        ++out.restore_objects;
                    }
                } else if (pages.host_resident(page)) {
                    const bool current = pages.cold_replica_current(page);
                    const bool spill   = !current && pages.can_spill(page);
                    if (first && current) { out.cold_bytes += stride; }
                    if (first && spill) {
                        out.write_bytes += stride;
                        ++out.write_objects;
                        ++writes[static_cast<std::size_t>(object_class(resource))];
                    }
                    if (first) { host_bytes += stride; }
                    if (!current && !spill) {
                        note(pages.cold_busy(page) ? ColdDemotionBlock::Busy
                                                   : ColdDemotionBlock::Unspillable);
                    } else if (pages.cold_busy(page) || pages.source_pins(page) != 0) {
                        note(pages.cold_busy(page) ? ColdDemotionBlock::Busy
                                                   : ColdDemotionBlock::Unreleasable);
                    } else if (pages.address_references(page) >
                               victim_holders[shared_page_key(pages, page, resource)]) {
                        // Another holder still uses this page: it stays in Host with its copy.
                    } else if (first) {
                        releasable += stride;
                        if (!pages.device_resident(page)) {
                            ++(main ? new_cold_main : new_cold_backend);
                        }
                    }
                } else if (pages.cold_replica_current(page)) {
                    if (first) { out.cold_bytes += stride; }
                } else {
                    if (first) { ++out.device_only_objects; }
                }
            });
        if (local == ColdDemotionBlock::None) {
            out.host_bytes_released_after_spill += releasable;
            out.host_bytes_retained += host_bytes - releasable;
        } else {
            out.host_bytes_retained += host_bytes;
            block(local);
        }
    }
    if (out.whole) {
        const auto exceeds = [](std::uint64_t added, std::uint32_t count, std::uint32_t limit) {
            return count > limit || added > limit - count;
        };
        if (exceeds(new_cold_states, state_store->cold_only_states(),
                    state_store->cold_only_limit()) ||
            exceeds(new_cold_main, text_kv_pages->cold_only_pages(),
                    text_kv_pages->cold_only_limit()) ||
            (backend_kv_pages && exceeds(new_cold_backend, backend_kv_pages->cold_only_pages(),
                                         backend_kv_pages->cold_only_limit()))) {
            block(ColdDemotionBlock::Unreleasable);
        } else if (!cold_store->can_allocate(writes)) {
            block(ColdDemotionBlock::Full);
        }
    }
    return out;
}

void ProgramImpl::submit_cold_objects(ColdJobId id) {
    auto& job = cold_jobs.at(id);
    while (!job.cancelled && job.next_submit < job.objects.size()) {
        auto& object  = job.objects[job.next_submit];
        object.ticket = next_cold_ticket_++;
        ColdSubmit submitted;
        if (job.kind == ColdJobKind::Spill) {
            submitted = cold_store->write(object.slot, object.memory, object.ticket);
        } else {
            const auto expected =
                object.resource == runtime::ContextResourceClass::State
                    ? state_store->cold_replica(object.state)->checksum
                    : object.pages->cold_replica(object.page)->checksum;
            submitted = cold_store->read(object.slot, object.memory, expected, object.ticket);
        }
        if (submitted == ColdSubmit::Busy) { return; }
        if (submitted == ColdSubmit::Disabled) {
            for (auto index = job.next_submit; index < job.objects.size(); ++index) {
                job.objects[index].finished = true;
                job.objects[index].status   = ColdIoStatus::IoError;
            }
            job.next_submit = job.objects.size();
            return;
        }
        object.submitted = true;
        ++job.outstanding;
        cold_tickets_[object.ticket] = {id, job.next_submit};
        ++job.next_submit;
    }
}

ColdStart ProgramImpl::start_spill(std::span<const CheckpointHandle> handles) {
    if (!cold_store || cold_store->disabled()) { return {.status = ColdStartStatus::Unavailable}; }
    for (const auto handle : handles) {
        if (!valid_checkpoint(handle)) { return {.status = ColdStartStatus::InvalidSource}; }
    }
    ColdJob job;
    job.kind = ColdJobKind::Spill;
    job.checkpoints.assign(handles.begin(), handles.end());
    ObjectSet seen;
    for (const auto handle : handles) {
        visit_checkpoint_objects(
            checkpoint(handle),
            [&](StateImageHandle state) {
                if (!seen.insert_state(*state_store, state) || !state_store->can_spill(state)) {
                    return;
                }
                job.objects.push_back(
                    {.resource = runtime::ContextResourceClass::State,
                     .state    = state,
                     .memory   = const_cast<std::byte*>(state_store->host_view(state).data),
                     .bytes    = state_images->host_layout().image_bytes,
                     .epoch    = state_store->content_epoch(state)});
            },
            [&](LogicalKVPageStore& pages, LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                if (!seen.insert_page(pages, page, resource) || !pages.can_spill(page)) { return; }
                const auto replica = pages.host_replica(page);
                const auto view =
                    host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                job.objects.push_back({.resource = resource,
                                       .pages    = &pages,
                                       .page     = page,
                                       .memory   = const_cast<std::byte*>(view.data()),
                                       .bytes    = cold_store->object_bytes(object_class(resource)),
                                       .epoch    = pages.content_epoch(page),
                                       .columns  = pages.committed_columns(page)});
            });
    }
    if (job.objects.empty()) { return {.status = ColdStartStatus::NothingToDo}; }
    ColdStart result{.status = ColdStartStatus::Started};
    for (std::size_t index = 0; index < job.objects.size(); ++index) {
        auto& object = job.objects[index];
        auto slot    = cold_store->allocate(object_class(object.resource));
        if (!slot) {
            for (std::size_t prior = 0; prior < index; ++prior) {
                cold_store->release(job.objects[prior].slot);
            }
            return {.status = cold_store->disabled() ? ColdStartStatus::Unavailable
                                                     : ColdStartStatus::Full};
        }
        object.slot = *slot;
        result.bytes += object.bytes;
    }
    for (const auto& object : job.objects) {
        if (object.resource == runtime::ContextResourceClass::State) {
            state_store->set_cold_busy(object.state, true);
        } else {
            object.pages->set_cold_busy(object.page, true);
        }
    }
    result.objects = static_cast<std::uint32_t>(job.objects.size());
    job.started    = std::chrono::steady_clock::now();
    const auto id  = next_cold_job_++;
    cold_jobs.emplace(id, std::move(job));
    result.job = id;
    submit_cold_objects(id);
    return result;
}

ColdStart ProgramImpl::start_restore(CheckpointHandle handle) {
    if (!cold_store || cold_store->disabled()) { return {.status = ColdStartStatus::Unavailable}; }
    if (!valid_checkpoint(handle)) { return {.status = ColdStartStatus::InvalidSource}; }
    if (const auto running = cold_restore_job(handle)) {
        return {.status = ColdStartStatus::Started, .job = running};
    }
    const auto& record = checkpoint(handle);
    std::optional<StateImageHandle> state;
    std::vector<LogicalKVPageHandle> main, backend;
    bool lost = false, busy = false;
    visit_checkpoint_objects(
        record,
        [&](StateImageHandle image) {
            if (!state_store->has_replica(image)) {
                lost = true;
            } else if (state_store->cold_only(image)) {
                if (state_store->cold_busy(image) || !state_store->cold_replica_current(image)) {
                    busy = true;
                } else {
                    state = image;
                }
            }
        },
        [&](const LogicalKVPageStore& pages, LogicalKVPageHandle page,
            runtime::ContextResourceClass resource) {
            if (!pages.has_replica(page)) {
                lost = true;
            } else if (pages.cold_only(page)) {
                if (pages.cold_busy(page) || pages.source_pins(page) ||
                    !pages.cold_replica_current(page)) {
                    busy = true;
                } else {
                    (resource == runtime::ContextResourceClass::MainKV ? main : backend)
                        .push_back(page);
                }
            }
        });
    if (lost) { return {.status = ColdStartStatus::InvalidSource}; }
    if (busy) { return {.status = ColdStartStatus::Busy}; }
    if (!state && main.empty() && backend.empty()) {
        return {.status = ColdStartStatus::NothingToDo};
    }
    ColdStart result;
    result.host_bytes = (state ? state_images->host_layout().image_bytes : 0) +
                        main.size() * text_host_kv_page_stride +
                        backend.size() * backend_host_kv_page_stride;
    ColdJob job;
    job.kind        = ColdJobKind::Restore;
    job.checkpoints = {handle};
    // The complete Host destination is reserved before any read starts (no streamed restore).
    if (state) {
        if (!state_store->reserve_cold_restore(*state)) {
            return {.status = ColdStartStatus::HostShortage, .host_bytes = result.host_bytes};
        }
        job.state_destination = state;
        job.objects.push_back({.resource = runtime::ContextResourceClass::State,
                               .state    = *state,
                               .slot     = state_store->cold_replica(*state)->slot,
                               .memory   = state_store->cold_restore_view(*state).data,
                               .bytes    = state_images->host_layout().image_bytes});
    }
    const auto reserve = [&](LogicalKVPageStore& pages, std::span<const LogicalKVPageHandle> list,
                             runtime::ContextResourceClass resource) {
        // A fragmented arena may still hold the destination as several extents.
        std::size_t begin = 0, run = list.size();
        while (begin < list.size()) {
            run           = std::min(run, list.size() - begin);
            auto reserved = host_kv_extents->prepare_cold_restore(pages, list.subspan(begin, run));
            if (!reserved) {
                if (run == 1) { return false; }
                run = (run + 1) / 2;
                continue;
            }
            job.extents.push_back(std::move(*reserved));
            const auto view = host_kv_extents->writable_view(job.extents.back());
            for (std::size_t offset = 0; offset < run; ++offset) {
                const auto page = list[begin + offset];
                job.objects.push_back(
                    {.resource = resource,
                     .pages    = &pages,
                     .page     = page,
                     .slot     = pages.cold_replica(page)->slot,
                     .memory   = view.subview(static_cast<std::uint32_t>(offset), 1).data(),
                     .bytes    = cold_store->object_bytes(object_class(resource))});
            }
            begin += run;
        }
        return true;
    };
    if (!reserve(*text_kv_pages, main, runtime::ContextResourceClass::MainKV) ||
        (backend_kv_pages &&
         !reserve(*backend_kv_pages, backend, runtime::ContextResourceClass::BackendKV))) {
        job.extents.clear();
        if (job.state_destination) { state_store->abort_cold_restore(*job.state_destination); }
        return {.status = ColdStartStatus::HostShortage, .host_bytes = result.host_bytes};
    }
    for (const auto& object : job.objects) {
        if (object.resource == runtime::ContextResourceClass::State) {
            state_store->set_cold_busy(object.state, true);
        } else {
            object.pages->set_cold_busy(object.page, true);
        }
        result.bytes += object.bytes;
    }
    ++checkpoints[handle.index].pins;
    job.pinned    = {handle};
    job.histories = {record.kv};
    job.started   = std::chrono::steady_clock::now();
    result.status  = ColdStartStatus::Started;
    result.objects = static_cast<std::uint32_t>(job.objects.size());
    const auto id  = next_cold_job_++;
    cold_jobs.emplace(id, std::move(job));
    result.job = id;
    submit_cold_objects(id);
    return result;
}

void ProgramImpl::finish_cold_job(ColdJobId id, std::vector<ColdCompletion>& out) {
    auto node = cold_jobs.extract(id);
    auto& job = node.mapped();
    ColdCompletion completion{.job         = id,
                              .kind        = job.kind,
                              .checkpoints = job.checkpoints,
                              .elapsed_ns  = static_cast<std::uint64_t>(
                                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - job.started)
                                      .count())};
    bool cancelled = job.cancelled, failed = false, corrupt = false;
    for (const auto& object : job.objects) {
        if (object.status == ColdIoStatus::Ok) {
            completion.bytes += object.bytes;
            ++completion.objects;
            continue;
        }
        cancelled |= object.status == ColdIoStatus::Cancelled;
        failed |= object.status == ColdIoStatus::IoError;
        corrupt |= object.status == ColdIoStatus::ChecksumMismatch;
        if (object.status != ColdIoStatus::Cancelled) { ++completion.failed_objects; }
    }
    // Every I/O has retired: release the per-object pins before publication or rollback.
    for (const auto& object : job.objects) {
        bool attached = false;
        if (object.resource == runtime::ContextResourceClass::State) {
            if (job.kind == ColdJobKind::Spill && object.status == ColdIoStatus::Ok &&
                state_store->content_epoch(object.state) == object.epoch) {
                state_store->attach_cold_replica(object.state, {.slot          = object.slot,
                                                                .content_epoch = object.epoch,
                                                                .checksum = object.checksum});
                attached = true;
            }
            state_store->set_cold_busy(object.state, false);
        } else {
            if (job.kind == ColdJobKind::Spill && object.status == ColdIoStatus::Ok &&
                object.pages->content_epoch(object.page) == object.epoch &&
                object.pages->committed_columns(object.page) == object.columns) {
                object.pages->attach_cold_replica(object.page,
                                                  {.slot              = object.slot,
                                                   .content_epoch     = object.epoch,
                                                   .committed_columns = object.columns,
                                                   .checksum          = object.checksum});
                attached = true;
            }
            object.pages->set_cold_busy(object.page, false);
        }
        if (job.kind == ColdJobKind::Spill && !attached) { cold_store->release(object.slot); }
    }
    if (job.kind == ColdJobKind::Spill) {
        // Sources whose last logical reference retired during the write can now release Host.
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    } else if (!cancelled && !failed && !corrupt) {
        for (auto& extent : job.extents) { (void)host_kv_extents->publish(std::move(extent)); }
        if (job.state_destination) { state_store->publish_cold_restore(*job.state_destination); }
    } else {
        job.extents.clear();
        if (job.state_destination) { state_store->abort_cold_restore(*job.state_destination); }
        // An unreadable or corrupt copy is never retried silently.
        for (const auto& object : job.objects) {
            if (object.status != ColdIoStatus::IoError &&
                object.status != ColdIoStatus::ChecksumMismatch) {
                continue;
            }
            if (object.resource == runtime::ContextResourceClass::State) {
                state_store->invalidate_cold_replica(object.state);
            } else {
                object.pages->invalidate_cold_replica(object.page);
            }
        }
        if (failed || corrupt) {
            for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
                const CheckpointHandle candidate{this, index, checkpoints[index].generation};
                if (valid_checkpoint(candidate) &&
                    cold_residency(candidate) == ColdResidency::Lost) {
                    completion.lost.push_back(candidate);
                }
            }
        }
    }
    job.extents.clear();
    for (const auto pinned : job.pinned) {
        if (!valid_checkpoint(pinned) || checkpoints[pinned.index].pins == 0) { std::terminate(); }
        --checkpoints[pinned.index].pins;
    }
    job.histories.clear();
    completion.status = corrupt     ? ColdJobStatus::ChecksumFailed
                        : failed    ? ColdJobStatus::Failed
                        : cancelled ? ColdJobStatus::Cancelled
                                    : ColdJobStatus::Completed;
    out.push_back(std::move(completion));
}

std::vector<ColdCompletion> ProgramImpl::poll_cold() {
    std::vector<ColdCompletion> out;
    if (!cold_store) { return out; }
    std::vector<ColdIoCompletion> completed;
    cold_store->poll(completed);
    for (const auto& io : completed) {
        const auto found = cold_tickets_.find(io.ticket);
        if (found == cold_tickets_.end()) {
            throw std::logic_error("cold completion has no Program job");
        }
        auto& job    = cold_jobs.at(found->second.first);
        auto& object = job.objects.at(found->second.second);
        cold_tickets_.erase(found);
        object.finished = true;
        object.status   = io.status;
        object.checksum = io.checksum;
        --job.outstanding;
    }
    std::vector<ColdJobId> finished;
    for (auto& [id, job] : cold_jobs) {
        submit_cold_objects(id);
        if (job.outstanding == 0 && job.next_submit == job.objects.size()) {
            finished.push_back(id);
        }
    }
    for (const auto id : finished) { finish_cold_job(id, out); }
    return out;
}

bool ProgramImpl::cancel_cold(ColdJobId id) {
    const auto found = cold_jobs.find(id);
    if (found == cold_jobs.end()) { return false; }
    auto& job     = found->second;
    job.cancelled = true;
    for (auto index = job.next_submit; index < job.objects.size(); ++index) {
        job.objects[index].finished = true;
        job.objects[index].status   = ColdIoStatus::Cancelled;
    }
    job.next_submit = job.objects.size();
    for (const auto& object : job.objects) {
        if (object.submitted && !object.finished) { (void)cold_store->cancel(object.ticket); }
    }
    return true;
}

std::size_t ProgramImpl::release_cold_host(std::span<const CheckpointHandle> handles) {
    if (!cold_store || !host_context_arena) { return 0; }
    const auto before = host_context_arena->occupied_bytes();

    // What one checkpoint would release, decided without changing anything. A checkpoint is
    // demoted whole or not at all: every object it holds in Host needs a current SSD copy and
    // every such Host replica must be releasable. Device-only and cold-only objects are not part
    // of what leaves Host. A page some other holder (a checkpoint outside the demoted set or a
    // running request) still references keeps its Host replica, with its SSD copy: releasing it
    // would make that holder read it back from the SSD.
    struct Plan {
        bool whole = true;
        std::optional<StateImageHandle> state;
        std::vector<HostKVPageReplicaRelease> releases;
        struct PageRef {
            LogicalKVPageStore* store;
            LogicalKVPageHandle page;
            runtime::ContextResourceClass resource;
        };
        std::vector<PageRef> pages;
    };
    const auto plan_of = [&](CheckpointHandle handle, auto& holders) {
        Plan plan;
        ObjectSet local;
        visit_checkpoint_objects(
            checkpoint(handle),
            [&](StateImageHandle image) {
                if (!plan.whole || !local.insert_state(*state_store, image) ||
                    state_store->cold_only(image)) {
                    return;
                }
                if (!state_store->host_resident(image)) { return; }
                // Another checkpoint outside the demoted set holds this image: it stays in Host
                // with its copy and does not count against the cold-only limit.
                if (state_store->checkpoint_references(image) >
                    holders[shared_state_key(*state_store, image)]) {
                    return;
                }
                if (!state_store->cold_replica_current(image) ||
                    !state_store->can_drop_host_to_cold(image)) {
                    plan.whole = false;
                } else {
                    plan.state = image;
                }
            },
            [&](LogicalKVPageStore& pages, LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                if (!plan.whole || !local.insert_page(pages, page, resource) ||
                    pages.cold_only(page)) {
                    return;
                }
                if (!pages.host_resident(page)) { return; }
                if (!pages.cold_replica_current(page) ||
                    !host_kv_extents->can_release_page_replica(pages, page)) {
                    plan.whole = false;
                    return;
                }
                if (pages.address_references(page) >
                    holders[shared_page_key(pages, page, resource)]) {
                    return;
                }
                plan.releases.push_back({&pages, page});
                plan.pages.push_back({&pages, page, resource});
            });
        return plan;
    };
    // A shared page is released only when every holder of it is demoted, so a holder that turns
    // out not to qualify withdraws its pages from the others: iterate to a fixed point.
    std::vector<CheckpointHandle> eligible;
    for (const auto handle : handles) {
        if (valid_checkpoint(handle)) { eligible.push_back(handle); }
    }
    std::vector<Plan> plans;
    for (;;) {
        auto holders = victim_page_holders(eligible);
        plans.clear();
        std::vector<CheckpointHandle> next;
        ObjectSet counted;
        std::uint32_t new_states = 0, new_main = 0, new_backend = 0;
        for (const auto handle : eligible) {
            auto plan = plan_of(handle, holders);
            std::uint32_t added_main = 0, added_backend = 0;
            std::uint32_t added_states = 0;
            if (plan.whole) {
                if (plan.state && !state_store->device_resident(*plan.state) &&
                    counted.insert_state(*state_store, *plan.state)) {
                    added_states = 1;
                }
                for (const auto& ref : plan.pages) {
                    if (ref.store->device_resident(ref.page) ||
                        !counted.insert_page(*ref.store, ref.page, ref.resource)) {
                        continue;
                    }
                    ++(ref.resource == runtime::ContextResourceClass::MainKV ? added_main : added_backend);
                }
                const auto fits = [](std::uint32_t total, std::uint32_t added,
                                     const LogicalKVPageStore* pages) {
                    return added == 0 || (pages && total + added <= pages->cold_only_limit() -
                                                                        pages->cold_only_pages());
                };
                plan.whole = new_states + added_states <=
                                 state_store->cold_only_limit() - state_store->cold_only_states() &&
                             fits(new_main, added_main, text_kv_pages.get()) &&
                             fits(new_backend, added_backend, backend_kv_pages.get());
            }
            if (plan.whole) {
                new_states += added_states;
                new_main += added_main;
                new_backend += added_backend;
                next.push_back(handle);
            }
            plans.push_back(std::move(plan));
        }
        if (next.size() == eligible.size()) { break; }
        eligible = std::move(next);
    }
    // Each shared page is released once, with the first holder that reaches it.
    for (auto& plan : plans) {
        if (!plan.whole) { continue; }
        std::erase_if(plan.releases, [&](const HostKVPageReplicaRelease& release) {
            return !release.pages->host_resident(release.page);
        });
        if (!plan.releases.empty() && !host_kv_extents->release_page_replicas(plan.releases)) {
            throw std::logic_error("cold Host release changed after its check");
        }
        if (plan.state && state_store->host_resident(*plan.state) &&
            !state_store->drop_host_to_cold(*plan.state)) {
            throw std::logic_error("cold StateImage release changed after its check");
        }
    }
    return before - host_context_arena->occupied_bytes();
}

std::uint64_t ProgramImpl::drop_cold(std::span<const CheckpointHandle> handles) {
    if (!cold_store) { return 0; }
    std::uint64_t released = 0;
    ObjectSet seen;
    for (const auto handle : handles) {
        if (!valid_checkpoint(handle)) { continue; }
        visit_checkpoint_objects(
            checkpoint(handle),
            [&](StateImageHandle state) {
                if (seen.insert_state(*state_store, state) &&
                    state_store->drop_cold_replica(state)) {
                    released += state_images->host_layout().image_bytes;
                }
            },
            [&](LogicalKVPageStore& pages, LogicalKVPageHandle page,
                runtime::ContextResourceClass resource) {
                if (seen.insert_page(pages, page, resource) && pages.drop_cold_replica(page)) {
                    released += cold_store->object_bytes(object_class(resource));
                }
            });
    }
    return released;
}

void ProgramImpl::shutdown_cold() noexcept {
    if (!cold_store) { return; }
    try {
        std::vector<ColdJobId> ids;
        for (const auto& [id, job] : cold_jobs) { ids.push_back(id); }
        for (const auto id : ids) { (void)cancel_cold(id); }
        cold_store->shutdown();
        while (!cold_jobs.empty()) {
            const auto before = cold_jobs.size();
            (void)poll_cold();
            if (cold_jobs.size() == before) { std::terminate(); }
        }
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen3_5::detail
