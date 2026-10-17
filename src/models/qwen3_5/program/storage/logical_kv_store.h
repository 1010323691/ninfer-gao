#pragma once

#include "core/cold_context_store.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class LogicalKVPageStore;
class HostKVExtentStore;

class HostKVExtentCapability {
public:
    HostKVExtentCapability() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(HostKVExtentCapability,
                                         HostKVExtentCapability) noexcept = default;

private:
    HostKVExtentCapability(const HostKVExtentStore* owner, std::uint32_t index,
                           std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const HostKVExtentStore* owner_ = nullptr;
    std::uint32_t index_            = 0;
    std::uint32_t generation_       = 0;

    friend class HostKVExtentStore;
};

// SSD copy of one full page. It is valid only for the content identity it was written from.
struct ColdPageReplica {
    ColdSlot slot;
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
    std::uint32_t checksum          = 0;
};

struct HostKVPageReplica {
    HostKVExtentCapability extent;
    std::uint32_t page_offset       = 0;
    std::uint32_t membership_node   = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
};

class LogicalKVPageHandle {
public:
    LogicalKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(LogicalKVPageHandle,
                                         LogicalKVPageHandle) noexcept = default;

private:
    LogicalKVPageHandle(const LogicalKVPageStore* owner, std::uint32_t index,
                        std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const LogicalKVPageStore* owner_ = nullptr;
    std::uint32_t index_             = 0;
    std::uint32_t generation_        = 0;

    friend class LogicalKVPageStore;
    friend class HostKVExtentStore;
};

class LogicalKVPageStore {
public:
    LogicalKVPageStore(DeviceKVPagePool& physical, std::uint32_t logical_capacity)
        : physical_(&physical), pages_(logical_capacity), free_(pages_.size()),
          free_count_(static_cast<std::uint32_t>(pages_.size())) {
        if (pages_.empty() || logical_capacity < physical.capacity_pages()) {
            throw std::invalid_argument("Logical KV page capacity is inconsistent");
        }
        for (std::uint32_t index = 0; index < pages_.size(); ++index) {
            free_[index] = static_cast<std::uint32_t>(pages_.size()) - 1U - index;
        }
        materialization_scratch_.reserve(physical.capacity_pages());
    }

    LogicalKVPageStore(const LogicalKVPageStore&)            = delete;
    LogicalKVPageStore& operator=(const LogicalKVPageStore&) = delete;
    LogicalKVPageStore(LogicalKVPageStore&&)                 = delete;
    LogicalKVPageStore& operator=(LogicalKVPageStore&&)      = delete;

    // The cold store releases slots of descriptors that retire or change content. Cold-only pages
    // (no Device or Host replica) are bounded by the descriptor allowance added for the tier.
    void attach_cold_store(ColdContextStore* store, std::uint32_t cold_only_limit) noexcept {
        cold_            = store;
        cold_only_limit_ = store ? cold_only_limit : 0;
    }

    [[nodiscard]] DeviceKVPagePool& physical_pool() noexcept { return *physical_; }

    [[nodiscard]] const DeviceKVPagePool& physical_pool() const noexcept { return *physical_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(pages_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] LogicalKVPageHandle materialize(DeviceKVPageReservation& reservation) {
        if (free_count_ == 0) {
            throw std::logic_error("logical KV descriptors exhausted before physical capacity");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch     = next_epoch(page.content_epoch);
        page.committed_columns = 0;
        page.references        = 1;
        page.active_references = 0;
        page.writer_references = 1;
        page.protected_columns = 0;
        page.occupied          = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void materialize(DeviceKVPageReservation& reservation,
                     std::span<LogicalKVPageHandle> destinations,
                     std::optional<LogicalKVPageHandle> preferred_predecessor = std::nullopt) {
        if (destinations.empty()) { return; }
        if (destinations.size() > free_count_ ||
            destinations.size() > materialization_scratch_.capacity()) {
            throw std::logic_error("logical KV batch materialization exceeds descriptor capacity");
        }
        for (const LogicalKVPageHandle destination : destinations) {
            if (destination.valid()) {
                throw std::logic_error("logical KV batch destination is already populated");
            }
        }
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            if (pages_[index].occupied) {
                throw std::logic_error("logical KV free descriptor is occupied");
            }
        }

        std::optional<DeviceKVPageHandle> physical_predecessor;
        if (preferred_predecessor) { physical_predecessor = physical(*preferred_predecessor); }
        materialization_scratch_.clear();
        physical_->materialize(reservation, static_cast<std::uint32_t>(destinations.size()),
                               materialization_scratch_, physical_predecessor);
        for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
            const std::uint32_t index = free_[free_count_ - 1U - offset];
            Page& page                = pages_[index];
            page.device_replica.emplace(std::move(materialization_scratch_[offset]));
            page.content_epoch     = next_epoch(page.content_epoch);
            page.committed_columns = 0;
            page.references        = 1;
            page.active_references = 0;
            page.writer_references = 1;
            page.protected_columns = 0;
            page.occupied          = true;
            destinations[offset]   = LogicalKVPageHandle(this, index, page.generation);
        }
        free_count_ -= static_cast<std::uint32_t>(destinations.size());
        materialization_scratch_.clear();
    }

    [[nodiscard]] LogicalKVPageHandle
    materialize_transfer_destination(DeviceKVPageReservation& reservation,
                                     std::uint32_t committed_columns) {
        if (committed_columns == 0 ||
            committed_columns > static_cast<std::uint32_t>(kPagedKVPageSize) || free_count_ == 0) {
            throw std::invalid_argument("logical KV transfer destination is invalid");
        }
        DeviceKVPageLease lease   = physical_->materialize_one(reservation);
        const std::uint32_t index = free_[--free_count_];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(lease));
        page.content_epoch      = next_epoch(page.content_epoch);
        page.committed_columns  = committed_columns;
        page.references         = 0;
        page.active_references  = 0;
        page.writer_references  = 0;
        page.protected_columns  = 0;
        page.destination_pinned = true;
        page.occupied           = true;
        return LogicalKVPageHandle(this, index, page.generation);
    }

    void publish_transfer_destination(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            std::terminate();
        }
        page.references         = 1;
        page.writer_references  = writer ? 1U : 0U;
        page.destination_pinned = false;
    }

    void abort_transfer_destination(LogicalKVPageHandle handle,
                                    DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.device_replica || page.pending_device_replica || page.host_replica ||
            !page.destination_pinned || page.references != 0 || page.writer_references != 0 ||
            page.source_pins != 0) {
            return;
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool valid(LogicalKVPageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < pages_.size() &&
               pages_[handle.index_].occupied &&
               pages_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] std::uint32_t descriptor_index(LogicalKVPageHandle handle) const {
        (void)require(handle);
        return handle.index_;
    }

    [[nodiscard]] DeviceKVPageHandle physical(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.device_replica) {
            throw std::logic_error("logical KV page has no Device replica");
        }
        return page.device_replica->handle();
    }

    [[nodiscard]] bool device_resident(LogicalKVPageHandle handle) const {
        return require(handle).device_replica.has_value();
    }

    [[nodiscard]] bool host_resident(LogicalKVPageHandle handle) const {
        return require(handle).host_replica.has_value();
    }

    [[nodiscard]] const HostKVPageReplica& host_replica(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        if (!page.host_replica) { throw std::logic_error("logical KV page has no Host replica"); }
        return *page.host_replica;
    }

    [[nodiscard]] std::uint64_t content_epoch(LogicalKVPageHandle handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t committed_columns(LogicalKVPageHandle handle) const {
        return require(handle).committed_columns;
    }

    [[nodiscard]] std::uint32_t address_references(LogicalKVPageHandle handle) const {
        return require(handle).references;
    }

    [[nodiscard]] std::uint32_t active_address_references(LogicalKVPageHandle handle) const {
        return require(handle).active_references;
    }

    [[nodiscard]] std::uint8_t writer_references(LogicalKVPageHandle handle) const {
        return require(handle).writer_references;
    }

    [[nodiscard]] std::uint32_t protected_columns(LogicalKVPageHandle handle) const {
        return require(handle).protected_columns;
    }

    [[nodiscard]] std::uint32_t source_pins(LogicalKVPageHandle handle) const {
        return require(handle).source_pins;
    }

    [[nodiscard]] bool can_pin_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 0 &&
               !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] bool can_pin_active_source(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica.has_value() && page.writer_references == 1 &&
               page.references == 1 && !page.destination_pinned &&
               page.source_pins != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] DeviceKVPageHandle reserve_device_replica(LogicalKVPageHandle handle,
                                                            DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.device_replica || page.pending_device_replica || !page.host_replica ||
            page.destination_pinned || page.source_pins != 0 ||
            page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV Device restore is not reservable");
        }
        page.pending_device_replica.emplace(physical_->materialize_one(reservation));
        page.destination_pinned = true;
        return page.pending_device_replica->handle();
    }

    void publish_device_replica(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.device_replica || !page.pending_device_replica || !page.destination_pinned ||
            !page.host_replica || page.host_replica->content_epoch != page.content_epoch ||
            page.host_replica->committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV Device restore is not publishable");
        }
        page.device_replica = std::move(page.pending_device_replica);
        page.pending_device_replica.reset();
        page.destination_pinned = false;
        sync_cold_only(page);
    }

    void abort_device_replica(LogicalKVPageHandle handle,
                              DeviceKVPageReservation& reservation) noexcept {
        if (!valid(handle)) { return; }
        Page& page = pages_[handle.index_];
        if (!page.pending_device_replica || !page.destination_pinned) { return; }
        physical_->dematerialize_one(reservation, std::move(*page.pending_device_replica));
        page.pending_device_replica.reset();
        page.destination_pinned = false;
    }

    [[nodiscard]] bool drop_device_replica(LogicalKVPageHandle handle) noexcept {
        if (!can_drop_device_replica(handle)) { return false; }
        Page& page = pages_[handle.index_];
        page.device_replica.reset();
        sync_cold_only(page);
        return true;
    }

    [[nodiscard]] bool host_replica_current(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
               page.host_replica->committed_columns == page.committed_columns;
    }

    [[nodiscard]] bool cold_replica_current(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.cold_replica && page.cold_replica->content_epoch == page.content_epoch &&
               page.cold_replica->committed_columns == page.committed_columns;
    }

    [[nodiscard]] const ColdPageReplica* cold_replica(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        return page.cold_replica ? &*page.cold_replica : nullptr;
    }

    [[nodiscard]] bool cold_busy(LogicalKVPageHandle handle) const {
        return require(handle).cold_busy;
    }

    // Only the SSD copy remains; restoring it to Host must precede any Device use.
    [[nodiscard]] bool cold_only(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        return !page.device_replica && !page.host_replica && page.cold_replica.has_value();
    }

    [[nodiscard]] bool has_replica(LogicalKVPageHandle handle) const {
        const Page& page = require(handle);
        return page.device_replica || page.host_replica || page.cold_replica;
    }

    [[nodiscard]] std::uint32_t cold_only_pages() const noexcept { return cold_only_count_; }

    [[nodiscard]] std::uint32_t cold_only_limit() const noexcept { return cold_only_limit_; }

    // A committed page whose Host replica is current can be copied to the SSD once, as a
    // whole-page object that records its committed columns. A partial tail qualifies only while
    // no writer can extend it: a checkpoint's tail is then immutable for that checkpoint. An
    // extension after the copy started changes the columns and the completion discards the copy.
    [[nodiscard]] bool can_spill(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle) || cold_ == nullptr) { return false; }
        const Page& page = pages_[handle.index_];
        return !page.cold_busy && !page.cold_replica && page.committed_columns != 0 &&
               (page.committed_columns == static_cast<std::uint32_t>(kPagedKVPageSize) ||
                page.writer_references == 0) &&
               host_replica_current(handle);
    }

    // One outstanding SSD transfer pins the page: its Host source (spill) or reserved Host
    // destination (restore) cannot be released, and its cold slot cannot change.
    void set_cold_busy(LogicalKVPageHandle handle, bool busy) {
        Page& page = require(handle);
        if (page.cold_busy == busy) { throw std::logic_error("logical KV cold pin is unbalanced"); }
        page.cold_busy = busy;
    }

    void attach_cold_replica(LogicalKVPageHandle handle, ColdPageReplica replica) {
        Page& page = require(handle);
        if (cold_ == nullptr || page.cold_replica || !replica.slot.valid() ||
            replica.content_epoch != page.content_epoch ||
            replica.committed_columns != page.committed_columns) {
            throw std::logic_error("logical KV cold replica is not publishable");
        }
        page.cold_replica = replica;
        sync_cold_only(page);
    }

    // Releases a redundant SSD copy. A cold-only page keeps its last replica.
    [[nodiscard]] bool drop_cold_replica(LogicalKVPageHandle handle) {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        if (!page.cold_replica || page.cold_busy || (!page.device_replica && !page.host_replica)) {
            return false;
        }
        release_cold(page);
        return true;
    }

    // An unreadable SSD copy is discarded even when it was the last replica; the page then has no
    // content and every checkpoint covering it is unrecoverable.
    void invalidate_cold_replica(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.cold_busy) { throw std::logic_error("busy cold replica cannot be invalidated"); }
        if (page.cold_replica) { release_cold(page); }
    }

    // Whether removing the Host replica keeps the page's content in another current replica.
    [[nodiscard]] bool host_release_preserves(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return !page.cold_busy &&
               (page.device_replica || page.references == 0 || cold_replica_current(handle));
    }

    [[nodiscard]] bool can_drop_device_replica(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.device_replica && host_replica_current(handle) &&
               !page.pending_device_replica && page.writer_references == 0 &&
               page.active_references == 0 && page.source_pins == 0 && !page.destination_pinned;
    }

    void retain_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == std::numeric_limits<std::uint32_t>::max() ||
            page.active_references >= page.references) {
            throw std::logic_error("logical KV active reference is not retainable");
        }
        ++page.active_references;
    }

    void release_active_reference(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.active_references == 0) {
            throw std::logic_error("logical KV page has no active reference");
        }
        --page.active_references;
    }

    void retain_reference(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (page.references == std::numeric_limits<std::uint32_t>::max() ||
            (writer && page.writer_references != 0)) {
            throw std::logic_error("logical KV page reference is not retainable");
        }
        ++page.references;
        if (writer) { page.writer_references = 1; }
    }

    [[nodiscard]] bool can_retain_reference(LogicalKVPageHandle handle,
                                            bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != std::numeric_limits<std::uint32_t>::max() &&
               (!writer || (page.writer_references == 0 && page.references == 0)) &&
               !page.destination_pinned;
    }

    void set_writer(LogicalKVPageHandle handle, bool writer) {
        Page& page = require(handle);
        if (writer && page.writer_references != 0) {
            throw std::logic_error("logical KV page already has a writer");
        }
        page.writer_references = writer ? 1U : 0U;
    }

    [[nodiscard]] bool can_set_writer(LogicalKVPageHandle handle, bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return writer ? page.writer_references == 0 && page.references == 1 &&
                            page.source_pins == 0 && page.device_replica.has_value()
                      : page.writer_references == 1;
    }

    void protect_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = std::max(page.protected_columns, columns);
    }

    void set_protected_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns) {
            throw std::invalid_argument("logical KV protection exceeds committed coverage");
        }
        page.protected_columns = columns;
    }

    void pin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("logical KV source pin count overflow");
        }
        ++page.source_pins;
    }

    void unpin_source(LogicalKVPageHandle handle) {
        Page& page = require(handle);
        if (page.source_pins == 0) { throw std::logic_error("logical KV source is not pinned"); }
        --page.source_pins;
    }

    void commit_coverage(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!page.device_replica || page.writer_references != 1 ||
            columns < page.committed_columns ||
            columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
            throw std::invalid_argument("logical KV committed coverage is not monotonic");
        }
        if (columns == page.committed_columns) { return; }
        page.committed_columns = columns;
        // An SSD copy of the shorter tail can never become current again. A copy still being
        // written is discarded by its completion instead.
        if (!page.cold_busy) { retire_stale_cold(page); }
    }

    void destructive_truncate(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (columns > page.committed_columns || columns < page.protected_columns ||
            page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica) {
            throw std::invalid_argument("logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
            retire_stale_cold(page);
        }
    }

    [[nodiscard]] bool can_destructive_truncate(LogicalKVPageHandle handle,
                                                std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && columns >= page.protected_columns &&
               page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_destructive_truncate_inactive(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool
    can_destructive_truncate_inactive_after_host_release(LogicalKVPageHandle handle,
                                                         std::uint32_t columns) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return columns <= page.committed_columns && page.references == 1 &&
               page.writer_references == 0 && page.source_pins == 0 && !page.destination_pinned &&
               page.device_replica.has_value();
    }

    void destructive_truncate_inactive(LogicalKVPageHandle handle, std::uint32_t columns) {
        Page& page = require(handle);
        if (!can_destructive_truncate_inactive(handle, columns)) {
            throw std::logic_error("inactive logical KV page is not destructively truncatable");
        }
        if (columns != page.committed_columns) {
            page.committed_columns = columns;
            page.content_epoch     = next_epoch(page.content_epoch);
            retire_stale_cold(page);
        }
        page.protected_columns = std::min(page.protected_columns, columns);
    }

    [[nodiscard]] bool can_dematerialize(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references == 1 && page.writer_references == 1 && page.source_pins == 0 &&
               !page.destination_pinned && !page.host_replica && page.device_replica.has_value();
    }

    [[nodiscard]] bool can_release_reference(LogicalKVPageHandle handle,
                                             bool writer) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        // An unrelated address may drop its alias while a transfer reads the source.
        // The final address reference remains responsible for keeping that source alive.
        if (page.references == 0 || (writer && page.writer_references != 1) ||
            page.active_references >= page.references ||
            (page.source_pins != 0 && page.references == 1) || page.destination_pinned) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_reference_after_active_reference(LogicalKVPageHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return page.references != 0 && page.active_references != 0 &&
               page.active_references <= page.references && page.writer_references <= 1 &&
               (page.source_pins == 0 || page.references > 1) && !page.destination_pinned;
    }

    [[nodiscard]] bool release_reference(LogicalKVPageHandle handle, bool writer) noexcept {
        if (!can_release_reference(handle, writer)) { return false; }
        Page& page = pages_[handle.index_];
        if (writer) { page.writer_references = 0; }
        if (--page.references != 0) { return true; }
        page.device_replica.reset();
        sync_cold_only(page);
        // A Host extent owns its page capability until the extent is released as one atomic
        // allocation. Keep the zero-reference logical descriptor alive long enough for
        // HostKVExtentStore to detach that replica; without a Host replica the descriptor can be
        // reclaimed immediately.
        if (!page.host_replica) { release_descriptor(handle, page); }
        return true;
    }

    [[nodiscard]] bool can_attach_host_replica(LogicalKVPageHandle handle, std::uint64_t epoch,
                                               std::uint32_t coverage) const noexcept {
        if (!valid(handle)) { return false; }
        const Page& page = pages_[handle.index_];
        return !page.host_replica && epoch == page.content_epoch &&
               coverage == page.committed_columns;
    }

    void attach_host_replica(LogicalKVPageHandle handle, HostKVPageReplica replica) {
        Page& page = require(handle);
        if (!replica.extent.valid() ||
            replica.membership_node == std::numeric_limits<std::uint32_t>::max() ||
            !can_attach_host_replica(handle, replica.content_epoch, replica.committed_columns)) {
            throw std::logic_error("logical KV Host replica is not publishable");
        }
        page.host_replica = replica;
        sync_cold_only(page);
    }

    void rebind_host_replica(LogicalKVPageHandle handle, HostKVPageReplica expected,
                             HostKVPageReplica replacement) noexcept {
        if (!valid(handle)) { std::terminate(); }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != expected.extent ||
            page.host_replica->page_offset != expected.page_offset ||
            page.host_replica->membership_node != expected.membership_node ||
            page.host_replica->content_epoch != expected.content_epoch ||
            page.host_replica->committed_columns != expected.committed_columns ||
            expected.content_epoch != page.content_epoch || !replacement.extent.valid() ||
            replacement.membership_node != expected.membership_node ||
            replacement.content_epoch != expected.content_epoch ||
            replacement.committed_columns != expected.committed_columns) {
            std::terminate();
        }
        page.host_replica = replacement;
    }

    [[nodiscard]] bool detach_host_replica(LogicalKVPageHandle handle,
                                           HostKVExtentCapability extent) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        if (!page.host_replica || page.host_replica->extent != extent ||
            !host_release_preserves(handle)) {
            return false;
        }
        page.host_replica.reset();
        sync_cold_only(page);
        if (page.references == 0 && page.source_pins == 0 && !page.device_replica) {
            release_descriptor(handle, page);
        }
        return true;
    }

    void dematerialize(LogicalKVPageHandle handle, DeviceKVPageReservation& reservation) {
        Page& page = require(handle);
        if (page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
            page.destination_pinned || page.host_replica || !page.device_replica) {
            throw std::logic_error("logical KV page is shared or has no Device replica");
        }
        physical_->dematerialize_one(reservation, std::move(*page.device_replica));
        page.device_replica.reset();
        release_descriptor(handle, page);
    }

    [[nodiscard]] bool release(LogicalKVPageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Page& page = pages_[handle.index_];
        return release_reference(handle, page.writer_references != 0);
    }

private:
    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t source_pins       = 0;
        std::uint8_t writer_references  = 0;
        bool destination_pinned         = false;
        bool occupied                   = false;
        std::optional<DeviceKVPageLease> device_replica;
        std::optional<DeviceKVPageLease> pending_device_replica;
        std::optional<HostKVPageReplica> host_replica;
        std::optional<ColdPageReplica> cold_replica;
        bool cold_busy         = false;
        bool counted_cold_only = false;
    };

    [[nodiscard]] static std::uint64_t next_epoch(std::uint64_t epoch) noexcept {
        ++epoch;
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Page& require(LogicalKVPageHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    [[nodiscard]] const Page& require(LogicalKVPageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index_];
    }

    void sync_cold_only(Page& page) noexcept {
        const bool now = page.occupied && !page.device_replica && !page.host_replica &&
                         page.cold_replica.has_value();
        if (now == page.counted_cold_only) { return; }
        page.counted_cold_only = now;
        if (now) {
            ++cold_only_count_;
        } else {
            --cold_only_count_;
        }
    }

    void release_cold(Page& page) noexcept {
        if (!page.cold_replica) { return; }
        if (page.cold_busy || cold_ == nullptr) { std::terminate(); }
        try {
            cold_->release(page.cold_replica->slot);
        } catch (...) { std::terminate(); }
        page.cold_replica.reset();
        sync_cold_only(page);
    }

    // Content changed: an SSD copy of the old identity can never become current again.
    void retire_stale_cold(Page& page) noexcept {
        if (page.cold_replica && (page.cold_replica->content_epoch != page.content_epoch ||
                                  page.cold_replica->committed_columns != page.committed_columns)) {
            release_cold(page);
        }
    }

    void release_descriptor(LogicalKVPageHandle handle, Page& page) noexcept {
        if (page.active_references != 0 || page.cold_busy) { std::terminate(); }
        release_cold(page);
        page.committed_columns  = 0;
        page.references         = 0;
        page.active_references  = 0;
        page.protected_columns  = 0;
        page.source_pins        = 0;
        page.writer_references  = 0;
        page.destination_pinned = false;
        page.occupied           = false;
        page.pending_device_replica.reset();
        page.host_replica.reset();
        page.occupied = false;
        sync_cold_only(page);
        if (++page.generation == 0) { ++page.generation; }
        free_[free_count_++] = handle.index_;
    }

    DeviceKVPagePool* physical_ = nullptr;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
    std::vector<DeviceKVPageLease> materialization_scratch_;
    std::uint32_t free_count_      = 0;
    ColdContextStore* cold_        = nullptr;
    std::uint32_t cold_only_limit_ = 0;
    std::uint32_t cold_only_count_ = 0;
};
} // namespace ninfer::models::qwen3_5::detail
