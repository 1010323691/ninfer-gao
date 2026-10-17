#include "core/cold_context_store.h"

#include "core/crc32c.h"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace ninfer {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kChunkBytes   = 8ULL << 20U;
constexpr std::size_t kBlockBytes   = ColdContextStore::kChecksumBlockBytes;
constexpr std::size_t kMinuteBuckets = 24U * 60U;
static_assert(kChunkBytes % kBlockBytes == 0);

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

[[noreturn]] void throw_errno(const std::string& what, int error) {
    throw std::system_error(error, std::generic_category(), what);
}

struct AlignedBuffer {
    AlignedBuffer() = default;

    AlignedBuffer(std::size_t bytes, std::size_t alignment) : size(bytes) {
        void* memory = nullptr;
        if (posix_memalign(&memory, alignment, std::max<std::size_t>(bytes, alignment)) != 0) {
            throw std::bad_alloc();
        }
        data = static_cast<std::byte*>(memory);
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
        }
        return *this;
    }

    ~AlignedBuffer() { std::free(data); }

    std::byte* data  = nullptr;
    std::size_t size = 0;
};

struct Op {
    std::uint64_t ticket = 0;
    ColdIoKind kind      = ColdIoKind::Write;
    int fd               = -1;
    std::uint64_t offset = 0;
    std::byte* memory    = nullptr;
    std::size_t bytes    = 0;
    std::uint32_t expected = 0;
    std::size_t chunks   = 0;
    std::size_t next     = 0;
    std::size_t running  = 0;
    std::size_t finished = 0;
    std::vector<std::uint32_t> blocks;
    std::uint64_t issued_bytes = 0;
    int error                  = 0;
    bool stop                  = false;
    bool cancel_requested      = false;
    bool queued                = false;
    bool done                  = false;
    bool internal              = false;
    ColdIoStatus status        = ColdIoStatus::Ok;
    std::uint32_t checksum     = 0;
    Clock::time_point submitted;
    Clock::time_point completed;
};

std::uint32_t digest(const std::vector<std::uint32_t>& blocks) {
    std::vector<std::byte> bytes(blocks.size() * 4);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const std::uint32_t value = blocks[index];
        for (std::size_t shift = 0; shift < 4; ++shift) {
            bytes[index * 4 + shift] = static_cast<std::byte>((value >> (8 * shift)) & 0xFFU);
        }
    }
    return crc32c(bytes.data(), bytes.size());
}

int write_all(int fd, const std::byte* data, std::size_t bytes, std::uint64_t offset) {
    while (bytes != 0) {
        const ssize_t written = ::pwrite(fd, data, bytes, static_cast<off_t>(offset));
        if (written < 0) {
            if (errno == EINTR) { continue; }
            return errno;
        }
        if (written == 0) { return EIO; }
        data += written;
        bytes -= static_cast<std::size_t>(written);
        offset += static_cast<std::uint64_t>(written);
    }
    return 0;
}

int read_all(int fd, std::byte* data, std::size_t bytes, std::uint64_t offset) {
    while (bytes != 0) {
        const ssize_t read = ::pread(fd, data, bytes, static_cast<off_t>(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            return errno;
        }
        if (read == 0) { return EIO; } // a short object is a lost replica
        data += read;
        bytes -= static_cast<std::size_t>(read);
        offset += static_cast<std::uint64_t>(read);
    }
    return 0;
}

} // namespace

const char* cold_object_class_name(ColdObjectClass object_class) noexcept {
    switch (object_class) {
    case ColdObjectClass::MainKVPage:
        return "main_kv_page";
    case ColdObjectClass::BackendKVPage:
        return "backend_kv_page";
    case ColdObjectClass::StateImage:
        return "state_image";
    }
    return "unknown";
}

const char* cold_io_status_name(ColdIoStatus status) noexcept {
    switch (status) {
    case ColdIoStatus::Ok:
        return "ok";
    case ColdIoStatus::Cancelled:
        return "cancelled";
    case ColdIoStatus::IoError:
        return "io_error";
    case ColdIoStatus::ChecksumMismatch:
        return "checksum_mismatch";
    }
    return "unknown";
}

std::uint32_t ColdContextStore::checksum(const std::byte* data, std::size_t bytes) {
    std::vector<std::uint32_t> blocks((bytes + kBlockBytes - 1) / kBlockBytes);
    crc32c_blocks(data, bytes, kBlockBytes, blocks);
    return digest(blocks);
}

struct ColdContextStore::Impl {
    struct Segment {
        ColdObjectClass object_class = ColdObjectClass::MainKVPage;
        int fd                       = -1;
        std::uint32_t generation     = 1;
        bool live                    = false;
        std::vector<std::uint8_t> used;
        std::vector<std::uint8_t> busy;
        std::vector<std::uint32_t> free;
        std::uint32_t occupied = 0;
        std::filesystem::path path;
    };

    struct Active {
        std::shared_ptr<Op> op;
        std::optional<ColdSlot> slot;
    };

    struct Minute {
        std::int64_t minute = -1;
        std::uint64_t bytes = 0;
    };

    explicit Impl(ColdStoreOptions selected) : options(std::move(selected)) {}

    ColdStoreOptions options;
    int lock_fd                = -1;
    std::size_t memory_alignment = 4096;
    std::size_t offset_alignment = 4096;
    std::array<std::uint64_t, kColdObjectClassCount> strides{};
    std::array<std::uint32_t, kColdObjectClassCount> slots_per_segment{};
    std::vector<Segment> segments;
    std::vector<std::uint32_t> free_entries;
    std::uint64_t segment_bytes  = 0;
    std::uint32_t live_segments  = 0;
    std::uint64_t next_segment_name = 0;
    std::array<std::uint64_t, kColdObjectClassCount> occupied_slots{};
    std::unordered_map<std::uint64_t, Active> active;
    std::array<Minute, kMinuteBuckets> minutes{};
    Clock::time_point epoch = Clock::now();
    ColdProbeResult probe;
    bool disabled_flag = false;
    std::string disabled_reason;
    std::uint32_t consecutive_failures = 0;
    std::uint64_t segments_created     = 0;
    std::uint64_t segments_unlinked    = 0;

    // Shared with I/O threads.
    mutable std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable idle_cv;
    std::deque<std::shared_ptr<Op>> reads;
    std::deque<std::shared_ptr<Op>> writes;
    std::vector<ColdIoCompletion> finished;
    std::size_t outstanding = 0;
    bool stopping           = false;
    std::vector<std::thread> workers;
    std::uint64_t write_count       = 0;
    std::uint64_t read_count        = 0;
    std::uint64_t write_bytes       = 0;
    std::uint64_t read_bytes        = 0;
    std::uint64_t probe_write_bytes = 0;
    std::uint64_t write_failures    = 0;
    std::uint64_t read_failures     = 0;
    std::uint64_t checksum_failures = 0;
    std::uint64_t cancelled         = 0;
    std::uint64_t inflight_write    = 0;
    std::uint64_t inflight_read     = 0;

    void disable(std::string reason) {
        if (disabled_flag) { return; }
        disabled_flag   = true;
        disabled_reason = std::move(reason);
    }

    // Requires mutex.
    void add_written_minute(std::uint64_t bytes) {
        const auto minute = std::chrono::duration_cast<std::chrono::minutes>(Clock::now() - epoch)
                                .count();
        auto& bucket = minutes[static_cast<std::size_t>(minute) % kMinuteBuckets];
        if (bucket.minute != minute) {
            bucket.minute = minute;
            bucket.bytes  = 0;
        }
        bucket.bytes += bytes;
    }

    // Requires mutex.
    [[nodiscard]] std::uint64_t written_24h() const {
        const auto minute = std::chrono::duration_cast<std::chrono::minutes>(Clock::now() - epoch)
                                .count();
        std::uint64_t total = 0;
        for (const auto& bucket : minutes) {
            if (bucket.minute >= 0 &&
                minute - bucket.minute < static_cast<std::int64_t>(kMinuteBuckets)) {
                total += bucket.bytes;
            }
        }
        return total;
    }

    const Segment& require(ColdSlot slot) const {
        if (!slot.valid() || slot.segment >= segments.size()) {
            throw std::invalid_argument("cold slot is invalid");
        }
        const Segment& segment = segments[slot.segment];
        if (!segment.live || segment.generation != slot.generation ||
            slot.index >= segment.used.size() || !segment.used[slot.index]) {
            throw std::invalid_argument("cold slot is stale");
        }
        return segment;
    }

    Segment& require(ColdSlot slot) {
        return const_cast<Segment&>(static_cast<const Impl&>(*this).require(slot));
    }

    [[nodiscard]] std::uint64_t file_offset(const Segment& segment, ColdSlot slot) const {
        return static_cast<std::uint64_t>(slot.index) *
               strides[static_cast<std::size_t>(segment.object_class)];
    }

    // Requires mutex.
    void finalize_locked(Op& op) {
        if (op.done) { return; }
        op.done      = true;
        op.completed = Clock::now();
        if (op.error) {
            op.status = ColdIoStatus::IoError;
        } else if (op.finished != op.chunks) {
            op.status = ColdIoStatus::Cancelled;
        } else {
            op.checksum = digest(op.blocks);
            op.status   = op.kind == ColdIoKind::Read && op.checksum != op.expected
                              ? ColdIoStatus::ChecksumMismatch
                              : ColdIoStatus::Ok;
        }
        auto& inflight = op.kind == ColdIoKind::Write ? inflight_write : inflight_read;
        inflight -= std::min<std::uint64_t>(inflight, op.bytes);
        if (op.kind == ColdIoKind::Write) {
            // Every issued byte counts, including the probe and partial failed or cancelled
            // writes, so the 24 h window and the lifetime total reconcile.
            write_bytes += op.issued_bytes;
            add_written_minute(op.issued_bytes);
            if (op.internal) { probe_write_bytes += op.issued_bytes; }
        } else {
            read_bytes += op.issued_bytes;
        }
        switch (op.status) {
        case ColdIoStatus::Ok:
            ++(op.kind == ColdIoKind::Write ? write_count : read_count);
            break;
        case ColdIoStatus::Cancelled:
            ++cancelled;
            break;
        case ColdIoStatus::IoError:
            ++(op.kind == ColdIoKind::Write ? write_failures : read_failures);
            break;
        case ColdIoStatus::ChecksumMismatch:
            ++checksum_failures;
            break;
        }
        if (!op.internal) {
            finished.push_back({.ticket     = op.ticket,
                                .kind       = op.kind,
                                .status     = op.status,
                                .error      = op.error,
                                .checksum   = op.checksum,
                                .bytes      = op.bytes,
                                .elapsed_ns = elapsed_ns(op.submitted, op.completed)});
        }
        --outstanding;
        idle_cv.notify_all();
    }

    int run_chunk(Op& op, std::size_t chunk, AlignedBuffer& bounce) {
        const std::size_t begin  = chunk * kChunkBytes;
        const std::size_t length = std::min(kChunkBytes, op.bytes - begin);
        std::byte* memory        = op.memory + begin;
        const std::uint64_t offset = op.offset + begin;
        if (options.fault_hook) {
            if (const int injected = options.fault_hook(op.kind, op.ticket)) { return injected; }
        }
        const bool direct = reinterpret_cast<std::uintptr_t>(memory) % memory_alignment == 0 &&
                            length % offset_alignment == 0 && offset % offset_alignment == 0;
        const std::size_t issued = direct ? length : round_up(length, offset_alignment);
        const std::span<std::uint32_t> blocks(op.blocks.data() + begin / kBlockBytes,
                                              (length + kBlockBytes - 1) / kBlockBytes);
        int error = 0;
        if (op.kind == ColdIoKind::Write) {
            crc32c_blocks(memory, length, kBlockBytes, blocks);
            if (direct) {
                error = write_all(op.fd, memory, length, offset);
            } else {
                std::memcpy(bounce.data, memory, length);
                std::memset(bounce.data + length, 0, issued - length);
                error = write_all(op.fd, bounce.data, issued, offset);
            }
        } else {
            if (direct) {
                error = read_all(op.fd, memory, length, offset);
            } else {
                error = read_all(op.fd, bounce.data, issued, offset);
                if (!error) { std::memcpy(memory, bounce.data, length); }
            }
            if (!error) { crc32c_blocks(memory, length, kBlockBytes, blocks); }
        }
        if (!error) {
            std::lock_guard lock(mutex);
            op.issued_bytes += issued;
        }
        return error;
    }

    void worker() {
        AlignedBuffer bounce(kChunkBytes, std::max<std::size_t>(memory_alignment, 4096));
        for (;;) {
            std::shared_ptr<Op> op;
            std::size_t chunk = 0;
            {
                std::unique_lock lock(mutex);
                work_cv.wait(lock, [&] { return stopping || !reads.empty() || !writes.empty(); });
                if (reads.empty() && writes.empty()) { return; }
                auto& queue = !reads.empty() ? reads : writes;
                op          = queue.front();
                if (op->stop) {
                    queue.pop_front();
                    op->queued = false;
                    if (op->running == 0) { finalize_locked(*op); }
                    continue;
                }
                chunk = op->next++;
                if (op->next == op->chunks) {
                    queue.pop_front();
                    op->queued = false;
                }
                ++op->running;
            }
            const int error = run_chunk(*op, chunk, bounce);
            std::lock_guard lock(mutex);
            --op->running;
            if (error) {
                if (!op->error) { op->error = error; }
                op->stop = true;
            } else {
                ++op->finished;
            }
            if (op->running == 0 && (op->finished == op->chunks || (op->stop && !op->queued))) {
                finalize_locked(*op);
            }
        }
    }

    void enqueue(const std::shared_ptr<Op>& op) {
        std::lock_guard lock(mutex);
        if (stopping) { throw std::logic_error("cold store is shut down"); }
        op->queued    = true;
        op->submitted = Clock::now();
        (op->kind == ColdIoKind::Read ? reads : writes).push_back(op);
        (op->kind == ColdIoKind::Read ? inflight_read : inflight_write) += op->bytes;
        ++outstanding;
        work_cv.notify_all();
    }

    std::shared_ptr<Op> make_op(ColdIoKind kind, int fd, std::uint64_t offset, std::byte* memory,
                                std::size_t bytes, std::uint64_t ticket) const {
        auto op    = std::make_shared<Op>();
        op->ticket = ticket;
        op->kind   = kind;
        op->fd     = fd;
        op->offset = offset;
        op->memory = memory;
        op->bytes  = bytes;
        op->chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
        op->blocks.assign((bytes + kBlockBytes - 1) / kBlockBytes, 0);
        return op;
    }

    void wait_internal(const std::shared_ptr<Op>& op) {
        std::unique_lock lock(mutex);
        idle_cv.wait(lock, [&] { return op->done; });
    }

    void run_probe() {
        const auto path = options.directory / "probe.bin";
        const int fd =
            ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
        if (fd < 0) { throw_errno("cold tier probe file " + path.string(), errno); }
        struct Cleanup {
            int fd;
            std::filesystem::path path;

            ~Cleanup() {
                ::close(fd);
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{fd, path};
        const std::uint64_t bytes = round_up(options.probe_bytes, offset_alignment);
        if (const int error = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes))) {
            throw_errno("cold tier probe preallocation", error);
        }
        AlignedBuffer buffer(bytes, std::max<std::size_t>(memory_alignment, 4096));
        std::uint64_t state = 0x9E3779B97F4A7C15ULL;
        for (std::size_t offset = 0; offset + 8 <= bytes; offset += 8) {
            state ^= state << 13U;
            state ^= state >> 7U;
            state ^= state << 17U;
            std::memcpy(buffer.data + offset, &state, 8);
        }
        auto write    = make_op(ColdIoKind::Write, fd, 0, buffer.data, bytes, 0);
        write->internal = true;
        enqueue(write);
        wait_internal(write);
        if (write->status != ColdIoStatus::Ok) {
            throw_errno("cold tier probe write", write->error ? write->error : EIO);
        }
        auto read      = make_op(ColdIoKind::Read, fd, 0, buffer.data, bytes, 0);
        read->internal = true;
        read->expected = write->checksum;
        enqueue(read);
        wait_internal(read);
        if (read->status != ColdIoStatus::Ok) {
            if (read->status == ColdIoStatus::ChecksumMismatch) {
                throw std::runtime_error("cold tier probe read back different bytes");
            }
            throw_errno("cold tier probe read", read->error ? read->error : EIO);
        }
        probe = {.bytes    = bytes,
                 .write_ns = elapsed_ns(write->submitted, write->completed),
                 .read_ns  = elapsed_ns(read->submitted, read->completed)};
        if (probe.write_bytes_per_second() < options.minimum_probe_bytes_per_second ||
            probe.read_bytes_per_second() < options.minimum_probe_bytes_per_second) {
            throw std::runtime_error(
                "cold tier device is slower than the configured minimum: write " +
                std::to_string(probe.write_bytes_per_second() / 1.0e6) + " MB/s, read " +
                std::to_string(probe.read_bytes_per_second() / 1.0e6) + " MB/s");
        }
    }

    void stop_workers() noexcept {
        {
            std::lock_guard lock(mutex);
            if (stopping && workers.empty()) { return; }
            stopping = true;
            for (auto* queue : {&reads, &writes}) {
                for (const auto& op : *queue) {
                    op->stop             = true;
                    op->cancel_requested = true;
                }
            }
        }
        work_cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) { worker.join(); }
        }
        workers.clear();
    }
};

ColdContextStore::ColdContextStore(ColdStoreOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
    auto& impl = *impl_;
    if (impl.options.directory.empty()) {
        throw std::invalid_argument("cold tier directory is empty");
    }
    if (!crc32c_supported()) {
        throw std::runtime_error("cold tier checksums require SSE4.2 CRC32C");
    }
    if (impl.options.io_threads == 0 || impl.options.capacity_bytes == 0 ||
        impl.options.segment_target_bytes == 0) {
        throw std::invalid_argument("cold tier capacity, segment size and I/O threads must be "
                                    "nonzero");
    }
    std::filesystem::create_directories(impl.options.directory);
    const auto lock_path = impl.options.directory / ".ninfer-cold.lock";
    impl.lock_fd         = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (impl.lock_fd < 0) { throw_errno("cold tier lock " + lock_path.string(), errno); }
    if (::flock(impl.lock_fd, LOCK_EX | LOCK_NB) != 0) {
        const int error = errno;
        ::close(impl.lock_fd);
        impl.lock_fd = -1;
        throw_errno("cold tier directory is used by another process", error);
    }
    try {
        // The tier is process-volatile: earlier segment files carry no recoverable state.
        for (const auto& entry : std::filesystem::directory_iterator(impl.options.directory)) {
            const auto name = entry.path().filename().string();
            if ((name.starts_with("seg_") && name.ends_with(".bin")) || name == "probe.bin") {
                std::filesystem::remove(entry.path());
            }
        }
        // A RAM-backed directory would only duplicate Host memory (and current tmpfs accepts
        // O_DIRECT, so the open check below does not exclude it).
        struct statfs kind {};
        if (::statfs(impl.options.directory.c_str(), &kind) != 0) {
            throw_errno("cold tier statfs", errno);
        }
        if (kind.f_type == TMPFS_MAGIC || kind.f_type == RAMFS_MAGIC) {
            throw std::runtime_error("cold tier directory is RAM-backed (tmpfs/ramfs): " +
                                     impl.options.directory.string());
        }
        // O_DIRECT support (some overlay and network file systems reject it).
        const auto check_path = impl.options.directory / "probe.bin";
        const int check_fd =
            ::open(check_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0600);
        if (check_fd < 0) {
            throw_errno("cold tier directory does not support O_DIRECT: " +
                            impl.options.directory.string(),
                        errno);
        }
#ifdef STATX_DIOALIGN
        struct statx details {};
        if (::statx(check_fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &details) == 0 &&
            (details.stx_mask & STATX_DIOALIGN) && details.stx_dio_offset_align != 0) {
            impl.memory_alignment = std::max<std::size_t>(details.stx_dio_mem_align, 1);
            impl.offset_alignment = details.stx_dio_offset_align;
        }
#endif
        ::close(check_fd);
        std::filesystem::remove(check_path);
        if (impl.offset_alignment > 4096 || 4096 % impl.offset_alignment != 0 ||
            impl.memory_alignment > 4096) {
            throw std::runtime_error("cold tier direct I/O alignment exceeds 4096 bytes");
        }
        struct statvfs fs {};
        if (::statvfs(impl.options.directory.c_str(), &fs) != 0) {
            throw_errno("cold tier statvfs", errno);
        }
        const auto available = static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize;
        if (available < impl.options.capacity_bytes + impl.options.free_space_margin_bytes) {
            throw std::runtime_error(
                "cold tier capacity " + std::to_string(impl.options.capacity_bytes) +
                " bytes plus margin exceeds available file system space " +
                std::to_string(available) + " bytes in " + impl.options.directory.string());
        }
        // Segments belong to one class and every used class needs live segments of its own, so
        // a small tier caps the target at an eighth of the capacity: three classes then still
        // fit several segments each instead of one class exhausting the tier with one file.
        const auto segment_target =
            std::min(impl.options.segment_target_bytes, impl.options.capacity_bytes / 8);
        for (std::size_t index = 0; index < kColdObjectClassCount; ++index) {
            const auto bytes = impl.options.object_bytes[index];
            if (bytes == 0) { continue; }
            impl.strides[index] = round_up(bytes, 4096);
            impl.slots_per_segment[index] = static_cast<std::uint32_t>(
                std::max<std::uint64_t>(1, segment_target / impl.strides[index]));
            if (impl.strides[index] * impl.slots_per_segment[index] > impl.options.capacity_bytes) {
                throw std::invalid_argument(std::string("cold tier capacity cannot hold one ") +
                                            cold_object_class_name(
                                                static_cast<ColdObjectClass>(index)) +
                                            " segment");
            }
        }
        for (std::uint32_t index = 0; index < impl.options.io_threads; ++index) {
            impl.workers.emplace_back([&impl] { impl.worker(); });
        }
        if (impl.options.probe_bytes) { impl.run_probe(); }
    } catch (...) {
        impl.stop_workers();
        ::close(impl.lock_fd);
        impl.lock_fd = -1;
        throw;
    }
}

ColdContextStore::~ColdContextStore() {
    auto& impl = *impl_;
    impl.stop_workers();
    for (auto& segment : impl.segments) {
        if (!segment.live) { continue; }
        ::close(segment.fd);
        std::error_code ignored;
        std::filesystem::remove(segment.path, ignored);
    }
    if (impl.lock_fd >= 0) {
        std::error_code ignored;
        std::filesystem::remove(impl.options.directory / ".ninfer-cold.lock", ignored);
        ::close(impl.lock_fd);
    }
}

std::size_t ColdContextStore::object_bytes(ColdObjectClass object_class) const noexcept {
    return impl_->options.object_bytes[static_cast<std::size_t>(object_class)];
}

std::uint64_t ColdContextStore::slot_stride(ColdObjectClass object_class) const noexcept {
    return impl_->strides[static_cast<std::size_t>(object_class)];
}

bool ColdContextStore::can_allocate(
    const std::array<std::uint64_t, kColdObjectClassCount>& objects) const noexcept {
    const auto& impl = *impl_;
    if (impl.disabled_flag) { return false; }
    std::uint64_t new_bytes = 0;
    for (std::size_t class_index = 0; class_index < kColdObjectClassCount; ++class_index) {
        if (!objects[class_index]) { continue; }
        const auto slots = impl.slots_per_segment[class_index];
        if (impl.options.object_bytes[class_index] == 0 || slots == 0) { return false; }
        std::uint64_t free = 0;
        for (const auto& segment : impl.segments) {
            if (segment.live && static_cast<std::size_t>(segment.object_class) == class_index) {
                free += segment.free.size();
            }
        }
        if (objects[class_index] <= free) { continue; }
        const auto segments = (objects[class_index] - free + slots - 1) / slots;
        new_bytes += segments * impl.strides[class_index] * slots;
    }
    return impl.segment_bytes + new_bytes <= impl.options.capacity_bytes;
}

std::optional<ColdSlot> ColdContextStore::allocate(ColdObjectClass object_class) {
    auto& impl              = *impl_;
    const auto class_index  = static_cast<std::size_t>(object_class);
    if (impl.disabled_flag || impl.options.object_bytes[class_index] == 0) { return std::nullopt; }
    // Densest live segment first, so sparse segments drain and their files can be unlinked.
    std::uint32_t chosen = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t index = 0; index < impl.segments.size(); ++index) {
        const auto& segment = impl.segments[index];
        if (!segment.live || segment.object_class != object_class || segment.free.empty()) {
            continue;
        }
        if (chosen == std::numeric_limits<std::uint32_t>::max() ||
            segment.free.size() < impl.segments[chosen].free.size()) {
            chosen = index;
        }
    }
    if (chosen == std::numeric_limits<std::uint32_t>::max()) {
        const auto slots = impl.slots_per_segment[class_index];
        const auto bytes = impl.strides[class_index] * slots;
        if (impl.segment_bytes + bytes > impl.options.capacity_bytes) { return std::nullopt; }
        struct statvfs fs {};
        if (::statvfs(impl.options.directory.c_str(), &fs) != 0 ||
            static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize <
                bytes + impl.options.free_space_margin_bytes) {
            return std::nullopt;
        }
        const auto path =
            impl.options.directory / ("seg_" + std::to_string(impl.next_segment_name++) + ".bin");
        const int fd =
            ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_DIRECT | O_CLOEXEC, 0600);
        if (fd < 0) { return std::nullopt; }
        if (::posix_fallocate(fd, 0, static_cast<off_t>(bytes)) != 0) {
            ::close(fd);
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            return std::nullopt;
        }
        if (!impl.free_entries.empty()) {
            chosen = impl.free_entries.back();
            impl.free_entries.pop_back();
        } else {
            chosen = static_cast<std::uint32_t>(impl.segments.size());
            impl.segments.emplace_back();
        }
        auto& segment        = impl.segments[chosen];
        segment.object_class = object_class;
        segment.fd           = fd;
        segment.live         = true;
        segment.path         = path;
        segment.used.assign(slots, 0);
        segment.busy.assign(slots, 0);
        segment.free.resize(slots);
        for (std::uint32_t slot = 0; slot < slots; ++slot) {
            segment.free[slot] = slots - 1 - slot;
        }
        segment.occupied = 0;
        impl.segment_bytes += bytes;
        ++impl.live_segments;
        ++impl.segments_created;
    }
    auto& segment              = impl.segments[chosen];
    const std::uint32_t slot   = segment.free.back();
    segment.free.pop_back();
    segment.used[slot] = 1;
    ++segment.occupied;
    ++impl.occupied_slots[class_index];
    return ColdSlot{.segment = chosen, .generation = segment.generation, .index = slot};
}

bool ColdContextStore::valid(ColdSlot slot) const noexcept {
    const auto& impl = *impl_;
    if (!slot.valid() || slot.segment >= impl.segments.size()) { return false; }
    const auto& segment = impl.segments[slot.segment];
    return segment.live && segment.generation == slot.generation &&
           slot.index < segment.used.size() && segment.used[slot.index];
}

ColdObjectClass ColdContextStore::object_class(ColdSlot slot) const {
    return impl_->require(slot).object_class;
}

void ColdContextStore::release(ColdSlot slot) {
    auto& impl    = *impl_;
    auto& segment = impl.require(slot);
    if (segment.busy[slot.index]) {
        throw std::logic_error("cold slot is released while its I/O is outstanding");
    }
    segment.used[slot.index] = 0;
    segment.free.push_back(slot.index);
    --segment.occupied;
    --impl.occupied_slots[static_cast<std::size_t>(segment.object_class)];
    if (segment.occupied != 0) { return; }
    // An empty segment returns its blocks to the file system (and its discard path) at once.
    ::close(segment.fd);
    std::error_code ignored;
    std::filesystem::remove(segment.path, ignored);
    const auto class_index = static_cast<std::size_t>(segment.object_class);
    impl.segment_bytes -= impl.strides[class_index] * impl.slots_per_segment[class_index];
    --impl.live_segments;
    ++impl.segments_unlinked;
    segment.fd   = -1;
    segment.live = false;
    segment.used.clear();
    segment.busy.clear();
    segment.free.clear();
    if (++segment.generation == 0) { ++segment.generation; }
    impl.free_entries.push_back(slot.segment);
}

ColdSubmit ColdContextStore::write(ColdSlot slot, const std::byte* source, std::uint64_t ticket) {
    auto& impl    = *impl_;
    auto& segment = impl.require(slot);
    if (impl.disabled_flag) { return ColdSubmit::Disabled; }
    if (segment.busy[slot.index] || impl.active.contains(ticket) || source == nullptr) {
        throw std::logic_error("cold write slot or ticket is already in use");
    }
    const auto bytes = impl.options.object_bytes[static_cast<std::size_t>(segment.object_class)];
    {
        std::lock_guard lock(impl.mutex);
        if (impl.inflight_write != 0 &&
            impl.inflight_write + bytes > impl.options.max_inflight_write_bytes) {
            return ColdSubmit::Busy;
        }
    }
    auto op = impl.make_op(ColdIoKind::Write, segment.fd, impl.file_offset(segment, slot),
                           const_cast<std::byte*>(source), bytes, ticket);
    segment.busy[slot.index] = 1;
    impl.active.emplace(ticket, Impl::Active{op, slot});
    impl.enqueue(op);
    return ColdSubmit::Submitted;
}

ColdSubmit ColdContextStore::read(ColdSlot slot, std::byte* destination,
                                  std::uint32_t expected_checksum, std::uint64_t ticket) {
    auto& impl    = *impl_;
    auto& segment = impl.require(slot);
    if (impl.disabled_flag) { return ColdSubmit::Disabled; }
    if (segment.busy[slot.index] || impl.active.contains(ticket) || destination == nullptr) {
        throw std::logic_error("cold read slot or ticket is already in use");
    }
    const auto bytes = impl.options.object_bytes[static_cast<std::size_t>(segment.object_class)];
    {
        std::lock_guard lock(impl.mutex);
        if (impl.inflight_read != 0 &&
            impl.inflight_read + bytes > impl.options.max_inflight_read_bytes) {
            return ColdSubmit::Busy;
        }
    }
    auto op      = impl.make_op(ColdIoKind::Read, segment.fd, impl.file_offset(segment, slot),
                                destination, bytes, ticket);
    op->expected = expected_checksum;
    segment.busy[slot.index] = 1;
    impl.active.emplace(ticket, Impl::Active{op, slot});
    impl.enqueue(op);
    return ColdSubmit::Submitted;
}

bool ColdContextStore::cancel(std::uint64_t ticket) {
    auto& impl  = *impl_;
    const auto found = impl.active.find(ticket);
    if (found == impl.active.end()) { return false; }
    std::lock_guard lock(impl.mutex);
    auto& op = *found->second.op;
    if (op.done) { return false; }
    op.cancel_requested = true;
    op.stop             = true;
    impl.work_cv.notify_all();
    return true;
}

void ColdContextStore::poll(std::vector<ColdIoCompletion>& out) {
    auto& impl = *impl_;
    std::vector<ColdIoCompletion> completed;
    {
        std::lock_guard lock(impl.mutex);
        completed.swap(impl.finished);
    }
    for (const auto& completion : completed) {
        const auto found = impl.active.find(completion.ticket);
        if (found == impl.active.end()) {
            throw std::logic_error("cold completion has no active operation");
        }
        if (found->second.slot) {
            auto& segment = impl.require(*found->second.slot);
            segment.busy[found->second.slot->index] = 0;
        }
        impl.active.erase(found);
        if (completion.status == ColdIoStatus::Ok) {
            impl.consecutive_failures = 0;
        } else if (completion.status == ColdIoStatus::IoError &&
                   ++impl.consecutive_failures >= impl.options.consecutive_failure_limit) {
            // Only failed I/O disables the tier. A checksum mismatch condemns that one copy:
            // every object of a corrupted checkpoint mismatches, which says nothing about
            // whether the device still accepts I/O.
            impl.disable(std::to_string(impl.consecutive_failures) +
                         " consecutive cold I/O failures; last " +
                         cold_io_status_name(completion.status) +
                         (completion.error
                              ? " (" + std::string(std::strerror(completion.error)) + ")"
                              : std::string()));
        }
        out.push_back(completion);
    }
    if (!impl.disabled_flag && !impl.active.empty()) {
        const auto now = Clock::now();
        for (const auto& [ticket, entry] : impl.active) {
            bool done = false;
            {
                std::lock_guard lock(impl.mutex);
                done = entry.op->done;
            }
            if (!done && now - entry.op->submitted > impl.options.io_timeout) {
                impl.disable("cold I/O exceeded its " +
                             std::to_string(impl.options.io_timeout.count()) + " ms timeout");
                break;
            }
        }
    }
}

bool ColdContextStore::idle() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->outstanding == 0;
}

void ColdContextStore::wait_idle() {
    std::unique_lock lock(impl_->mutex);
    impl_->idle_cv.wait(lock, [&] { return impl_->outstanding == 0; });
}

void ColdContextStore::shutdown() noexcept {
    impl_->stop_workers();
    impl_->disable("cold tier is shut down");
}

ColdStoreStats ColdContextStore::stats() const {
    const auto& impl = *impl_;
    ColdStoreStats out;
    out.capacity_bytes    = impl.options.capacity_bytes;
    out.segment_bytes     = impl.segment_bytes;
    out.segments          = impl.live_segments;
    out.occupied_slots    = impl.occupied_slots;
    for (std::size_t index = 0; index < kColdObjectClassCount; ++index) {
        out.occupied_bytes += impl.occupied_slots[index] * impl.options.object_bytes[index];
    }
    out.segments_created  = impl.segments_created;
    out.segments_unlinked = impl.segments_unlinked;
    out.disabled          = impl.disabled_flag;
    std::lock_guard lock(impl.mutex);
    out.write_bytes_24h      = impl.written_24h();
    out.write_count          = impl.write_count;
    out.read_count           = impl.read_count;
    out.write_bytes          = impl.write_bytes;
    out.read_bytes           = impl.read_bytes;
    out.probe_write_bytes    = impl.probe_write_bytes;
    out.write_failures       = impl.write_failures;
    out.read_failures        = impl.read_failures;
    out.checksum_failures    = impl.checksum_failures;
    out.cancelled            = impl.cancelled;
    out.inflight_write_bytes = impl.inflight_write;
    out.inflight_read_bytes  = impl.inflight_read;
    return out;
}

const ColdProbeResult& ColdContextStore::probe() const noexcept { return impl_->probe; }

bool ColdContextStore::disabled() const noexcept { return impl_->disabled_flag; }

std::string ColdContextStore::disabled_reason() const { return impl_->disabled_reason; }

const std::filesystem::path& ColdContextStore::directory() const noexcept {
    return impl_->options.directory;
}

std::size_t ColdContextStore::direct_io_memory_alignment() const noexcept {
    return impl_->memory_alignment;
}

std::size_t ColdContextStore::direct_io_offset_alignment() const noexcept {
    return impl_->offset_alignment;
}

std::filesystem::path ColdContextStore::segment_path(ColdSlot slot) const {
    return impl_->require(slot).path;
}

std::uint64_t ColdContextStore::slot_offset(ColdSlot slot) const {
    const auto& segment = impl_->require(slot);
    return impl_->file_offset(segment, slot);
}

} // namespace ninfer
