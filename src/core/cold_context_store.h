#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninfer {

// Fixed-size byte objects of the SSD context tier. The store never interprets their contents.
enum class ColdObjectClass : std::uint8_t { MainKVPage, BackendKVPage, StateImage };
inline constexpr std::size_t kColdObjectClassCount = 3;

// One fixed slot in a segment file. The segment generation rejects a slot after its segment was
// unlinked and its table entry reused.
struct ColdSlot {
    std::uint32_t segment    = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t generation = 0;
    std::uint32_t index      = 0;

    [[nodiscard]] bool valid() const noexcept {
        return segment != std::numeric_limits<std::uint32_t>::max();
    }

    friend bool operator==(ColdSlot, ColdSlot) noexcept = default;
};

enum class ColdIoKind : std::uint8_t { Write, Read };
enum class ColdIoStatus : std::uint8_t { Ok, Cancelled, IoError, ChecksumMismatch };

struct ColdIoCompletion {
    std::uint64_t ticket = 0;
    ColdIoKind kind      = ColdIoKind::Write;
    ColdIoStatus status  = ColdIoStatus::Ok;
    int error            = 0; // errno for IoError
    // Write: digest of the bytes written. Read: digest of the bytes read.
    std::uint32_t checksum  = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t elapsed_ns = 0;
};

// Testing seam: returning a nonzero errno fails that chunk before it reaches the file.
using ColdIoFaultHook = std::function<int(ColdIoKind, std::uint64_t ticket)>;

struct ColdStoreOptions {
    std::filesystem::path directory;
    std::uint64_t capacity_bytes = 0;
    // Bytes of one object per class; zero leaves that class unused.
    std::array<std::size_t, kColdObjectClassCount> object_bytes{};
    std::uint64_t segment_target_bytes = 1ULL << 30U;
    std::uint32_t io_threads           = 4;
    std::uint64_t max_inflight_write_bytes = 2ULL << 30U;
    std::uint64_t max_inflight_read_bytes  = 4ULL << 30U;
    // Startup sequential write+read probe through the production I/O path. Zero skips it.
    std::uint64_t probe_bytes                   = 512ULL << 20U;
    double minimum_probe_bytes_per_second       = 500.0e6;
    std::uint32_t consecutive_failure_limit     = 3;
    std::chrono::milliseconds io_timeout        = std::chrono::milliseconds(30000);
    // Filesystem space that must remain available beyond the configured capacity at startup and
    // beyond each new segment.
    std::uint64_t free_space_margin_bytes = 1ULL << 30U;
    ColdIoFaultHook fault_hook;
};

struct ColdProbeResult {
    std::uint64_t bytes    = 0;
    std::uint64_t write_ns = 0;
    std::uint64_t read_ns  = 0;

    [[nodiscard]] double write_bytes_per_second() const noexcept {
        return write_ns ? static_cast<double>(bytes) * 1.0e9 / static_cast<double>(write_ns) : 0.0;
    }

    [[nodiscard]] double read_bytes_per_second() const noexcept {
        return read_ns ? static_cast<double>(bytes) * 1.0e9 / static_cast<double>(read_ns) : 0.0;
    }
};

struct ColdStoreStats {
    std::uint64_t capacity_bytes = 0;
    std::uint64_t segment_bytes  = 0; // allocated segment files
    std::uint64_t occupied_bytes = 0; // object payload bytes in occupied slots
    std::uint32_t segments       = 0;
    std::array<std::uint64_t, kColdObjectClassCount> occupied_slots{};
    std::uint64_t write_count        = 0;
    std::uint64_t read_count         = 0;
    // Bytes issued to the file system, including the startup probe and alignment padding.
    std::uint64_t write_bytes        = 0;
    std::uint64_t read_bytes         = 0;
    std::uint64_t write_bytes_24h    = 0;
    std::uint64_t probe_write_bytes  = 0;
    std::uint64_t write_failures     = 0;
    std::uint64_t read_failures      = 0;
    std::uint64_t checksum_failures  = 0;
    std::uint64_t cancelled          = 0;
    std::uint64_t inflight_write_bytes = 0;
    std::uint64_t inflight_read_bytes  = 0;
    std::uint64_t segments_created   = 0;
    std::uint64_t segments_unlinked  = 0;
    bool disabled                    = false;
};

enum class ColdSubmit : std::uint8_t { Submitted, Busy, Disabled };

// Process-volatile SSD tier of fixed-size slots grouped by object class into segment files.
// The constructor locks and clears the directory, checks O_DIRECT support and capacity, and
// measures the device. Slots are written once after allocation and only reused after release.
// Single-owner contract: allocate/release/write/read/cancel/poll/stats are called from one control
// thread. I/O threads touch only submitted memory and never call back into the owner.
class ColdContextStore {
public:
    explicit ColdContextStore(ColdStoreOptions options);
    ~ColdContextStore();

    ColdContextStore(const ColdContextStore&)            = delete;
    ColdContextStore& operator=(const ColdContextStore&) = delete;

    [[nodiscard]] std::size_t object_bytes(ColdObjectClass object_class) const noexcept;
    [[nodiscard]] std::uint64_t slot_stride(ColdObjectClass object_class) const noexcept;

    // nullopt when capacity, file system space, or the Disabled state prevents allocation.
    [[nodiscard]] std::optional<ColdSlot> allocate(ColdObjectClass object_class);
    // Read-only quote: whether free slots plus new segments within the capacity hold `objects`
    // per class. It does not check file system space, so allocate() may still fail.
    [[nodiscard]] bool can_allocate(
        const std::array<std::uint64_t, kColdObjectClassCount>& objects) const noexcept;
    [[nodiscard]] bool valid(ColdSlot slot) const noexcept;
    [[nodiscard]] ColdObjectClass object_class(ColdSlot slot) const;
    // A slot with outstanding I/O cannot be released; empty segments are unlinked immediately.
    void release(ColdSlot slot);

    // Asynchronous whole-object I/O. Memory stays owned, valid and (for writes) unmodified by the
    // caller until the ticket's completion is returned by poll(), including after cancel().
    [[nodiscard]] ColdSubmit write(ColdSlot slot, const std::byte* source, std::uint64_t ticket);
    [[nodiscard]] ColdSubmit read(ColdSlot slot, std::byte* destination,
                                  std::uint32_t expected_checksum, std::uint64_t ticket);
    // Requests cancellation; unstarted chunks are skipped and the completion reports Cancelled
    // unless every chunk had already finished.
    bool cancel(std::uint64_t ticket);
    // Appends finished completions; never blocks.
    void poll(std::vector<ColdIoCompletion>& out);
    [[nodiscard]] bool idle() const noexcept;
    // Blocks until every submitted operation finished; completions remain queued for poll().
    void wait_idle();
    // Cancels queued work, waits for running chunks and stops the I/O threads.
    void shutdown() noexcept;

    [[nodiscard]] ColdStoreStats stats() const;
    [[nodiscard]] const ColdProbeResult& probe() const noexcept;
    [[nodiscard]] bool disabled() const noexcept;
    [[nodiscard]] std::string disabled_reason() const;
    [[nodiscard]] const std::filesystem::path& directory() const noexcept;
    [[nodiscard]] std::size_t direct_io_memory_alignment() const noexcept;
    [[nodiscard]] std::size_t direct_io_offset_alignment() const noexcept;

    // Physical location, for diagnostics and fault-injection tests.
    [[nodiscard]] std::filesystem::path segment_path(ColdSlot slot) const;
    [[nodiscard]] std::uint64_t slot_offset(ColdSlot slot) const;

    // Digest stored with a replica: CRC-32C over the little-endian CRC-32C values of each 1 MiB
    // block of the object.
    static constexpr std::size_t kChecksumBlockBytes = 1ULL << 20U;
    [[nodiscard]] static std::uint32_t checksum(const std::byte* data, std::size_t bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* cold_object_class_name(ColdObjectClass object_class) noexcept;
[[nodiscard]] const char* cold_io_status_name(ColdIoStatus status) noexcept;

} // namespace ninfer
