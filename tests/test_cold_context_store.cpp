#include "core/cold_context_store.h"
#include "core/crc32c.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/statfs.h>
#include <unistd.h>

namespace {

using namespace ninfer;
using namespace std::chrono_literals;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// Sizes mirror the real fp8 objects: an aligned Main page, an unaligned Backend page and a
// multi-chunk StateImage whose tail is not a multiple of 4096.
constexpr std::size_t kMain    = 2'113'536;
constexpr std::size_t kBackend = 132'096;
constexpr std::size_t kState   = (17ULL << 20U) + 512;

std::filesystem::path g_root;

ColdStoreOptions options(const std::string& name) {
    ColdStoreOptions out;
    out.directory              = g_root / name;
    out.capacity_bytes         = 96ULL << 20U;
    out.object_bytes           = {kMain, kBackend, kState};
    out.segment_target_bytes   = 16ULL << 20U;
    out.io_threads             = 2;
    out.probe_bytes            = 0;
    out.free_space_margin_bytes = 64ULL << 20U;
    return out;
}

std::vector<std::byte> pattern(std::size_t bytes, std::uint64_t seed) {
    std::vector<std::byte> out(bytes);
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (auto& value : out) {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        value = static_cast<std::byte>(state & 0xFFU);
    }
    return out;
}

std::vector<ColdIoCompletion> drain(ColdContextStore& store, std::size_t expected) {
    std::vector<ColdIoCompletion> out;
    const auto deadline = std::chrono::steady_clock::now() + 30s;
    while (out.size() < expected && std::chrono::steady_clock::now() < deadline) {
        store.poll(out);
        if (out.size() < expected) { std::this_thread::sleep_for(1ms); }
    }
    if (out.size() != expected) { throw std::runtime_error("cold completions did not arrive"); }
    return out;
}

std::size_t segment_files(const std::filesystem::path& directory) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        count += name.starts_with("seg_") ? 1U : 0U;
    }
    return count;
}

void test_crc32c() {
    const std::string_view text = "123456789";
    expect(crc32c(reinterpret_cast<const std::byte*>(text.data()), text.size()) == 0xE3069283U,
           "CRC-32C check value");
    const auto data = pattern((5ULL << 20U) + 77, 7);
    std::vector<std::uint32_t> blocks(6);
    crc32c_blocks(data.data(), data.size(), 1ULL << 20U, blocks);
    bool same = true;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto begin  = index << 20U;
        const auto length = std::min<std::size_t>(1ULL << 20U, data.size() - begin);
        same &= blocks[index] == crc32c(data.data() + begin, length);
    }
    expect(same, "interleaved block CRCs equal independent CRCs");
}

void test_rejects_tmpfs() {
    struct statfs fs {};
    if (::statfs("/dev/shm", &fs) != 0 || fs.f_type != TMPFS_MAGIC) { return; }
    auto selected      = options("unused");
    selected.directory = "/dev/shm/ninfer-cold-tier-test";
    bool rejected      = false;
    try {
        ColdContextStore store(selected);
    } catch (const std::exception&) { rejected = true; }
    std::error_code ignored;
    std::filesystem::remove_all(selected.directory, ignored);
    expect(rejected, "a RAM-backed tmpfs directory is rejected as a cold tier");
}

void test_round_trip_and_probe() {
    auto selected                          = options("roundtrip");
    selected.probe_bytes                   = 32ULL << 20U;
    selected.minimum_probe_bytes_per_second = 1.0e6;
    ColdContextStore store(selected);
    expect(store.probe().bytes == (32ULL << 20U) && store.probe().write_ns > 0 &&
               store.probe().read_ns > 0,
           "startup probe measured both directions");
    std::cout << "dio alignment memory=" << store.direct_io_memory_alignment()
              << " offset=" << store.direct_io_offset_alignment() << " probe write "
              << store.probe().write_bytes_per_second() / 1.0e9 << " GB/s read "
              << store.probe().read_bytes_per_second() / 1.0e9 << " GB/s\n";
    // Host arena allocations are only 256-byte aligned; exercise both direct and bounce paths.
    struct Object {
        ColdObjectClass object_class;
        std::size_t bytes;
        std::size_t misalignment;
    };
    const std::vector<Object> objects{{ColdObjectClass::MainKVPage, kMain, 0},
                                      {ColdObjectClass::MainKVPage, kMain, 256},
                                      {ColdObjectClass::BackendKVPage, kBackend, 256},
                                      {ColdObjectClass::StateImage, kState, 0},
                                      {ColdObjectClass::StateImage, kState, 256}};
    std::vector<std::unique_ptr<std::byte[]>> storage;
    std::vector<std::byte*> sources, destinations;
    std::vector<ColdSlot> slots;
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const auto& object = objects[index];
        std::byte* source      = storage.emplace_back(new std::byte[object.bytes + 8192]).get();
        std::byte* destination = storage.emplace_back(new std::byte[object.bytes + 8192]).get();
        const auto align       = [&](std::byte* base) {
            return reinterpret_cast<std::byte*>(
                       (reinterpret_cast<std::uintptr_t>(base) + 4095) & ~std::uintptr_t{4095}) +
                   object.misalignment;
        };
        auto* src = align(source);
        auto* dst = align(destination);
        const auto data = pattern(object.bytes, index + 11);
        std::memcpy(src, data.data(), object.bytes);
        std::memset(dst, 0xA5, object.bytes);
        sources.push_back(src);
        destinations.push_back(dst);
        auto slot = store.allocate(object.object_class);
        expect(slot.has_value(), "slot allocation within capacity");
        slots.push_back(*slot);
        expect(store.write(*slot, src, 100 + index) == ColdSubmit::Submitted, "write submitted");
    }
    auto written = drain(store, objects.size());
    std::vector<std::uint32_t> checksums(objects.size());
    for (const auto& completion : written) {
        const auto index = completion.ticket - 100;
        expect(completion.status == ColdIoStatus::Ok && completion.kind == ColdIoKind::Write,
               "write completed");
        expect(completion.checksum ==
                   ColdContextStore::checksum(sources[index], objects[index].bytes),
               "write digest equals the reference digest of the source bytes");
        checksums[index] = completion.checksum;
    }
    for (std::size_t index = 0; index < objects.size(); ++index) {
        expect(store.read(slots[index], destinations[index], checksums[index], 200 + index) ==
                   ColdSubmit::Submitted,
               "read submitted");
    }
    for (const auto& completion : drain(store, objects.size())) {
        const auto index = completion.ticket - 200;
        expect(completion.status == ColdIoStatus::Ok, "read verified");
        expect(std::memcmp(sources[index], destinations[index], objects[index].bytes) == 0,
               "read bytes equal written bytes");
    }
    const auto stats = store.stats();
    expect(stats.occupied_slots[0] == 2 && stats.occupied_slots[1] == 1 &&
               stats.occupied_slots[2] == 2,
           "occupied slots per class");
    expect(stats.write_bytes >= stats.probe_write_bytes + 2 * kMain + kBackend + 2 * kState,
           "issued write bytes include probe and padded objects");
    expect(stats.write_bytes_24h == stats.write_bytes,
           "the 24 h write window holds every issued write byte of a young store");
    for (const auto slot : slots) { store.release(slot); }
    expect(segment_files(selected.directory) == 0 && store.stats().segments == 0,
           "empty segments are unlinked");
    expect(!store.valid(slots.front()), "released slot is stale");
}

void test_corruption_and_lost_file() {
    auto selected = options("corrupt");
    ColdContextStore store(selected);
    const auto data = pattern(kMain, 3);
    auto slot       = *store.allocate(ColdObjectClass::MainKVPage);
    auto other      = *store.allocate(ColdObjectClass::MainKVPage);
    expect(store.write(slot, data.data(), 1) == ColdSubmit::Submitted &&
               store.write(other, data.data(), 2) == ColdSubmit::Submitted,
           "corruption fixture writes");
    const auto written = drain(store, 2);
    // Flip one bit in the middle of the first object through an ordinary descriptor.
    {
        const int fd = ::open(store.segment_path(slot).c_str(), O_RDWR);
        std::byte value{};
        const auto offset = static_cast<off_t>(store.slot_offset(slot) + kMain / 2);
        expect(::pread(fd, &value, 1, offset) == 1, "read byte to corrupt");
        value ^= std::byte{0x10};
        expect(::pwrite(fd, &value, 1, offset) == 1, "flip one bit");
        ::fsync(fd);
        ::close(fd);
    }
    std::vector<std::byte> destination(kMain);
    expect(store.read(slot, destination.data(), written[0].ticket == 1 ? written[0].checksum
                                                                       : written[1].checksum,
                      3) == ColdSubmit::Submitted,
           "corrupt read submitted");
    auto read = drain(store, 1);
    expect(read[0].status == ColdIoStatus::ChecksumMismatch, "single-bit flip is detected");
    const auto expected_digest = written[0].ticket == 1 ? written[0].checksum : written[1].checksum;
    for (std::uint64_t ticket = 10; ticket < 13; ++ticket) {
        expect(store.read(slot, destination.data(), expected_digest, ticket) ==
                   ColdSubmit::Submitted,
               "repeated corrupt read submitted");
        (void)drain(store, 1);
    }
    expect(!store.disabled(), "checksum mismatches condemn copies, not the tier");
    // A segment file truncated by an external actor reads short and reports an I/O error.
    expect(::truncate(store.segment_path(other).c_str(),
                      static_cast<off_t>(store.slot_offset(other) + 4096)) == 0,
           "truncate segment");
    expect(store.read(other, destination.data(), 0, 4) == ColdSubmit::Submitted,
           "lost read submitted");
    read = drain(store, 1);
    expect(read[0].status == ColdIoStatus::IoError, "truncated segment is a failed read");
    expect(store.stats().checksum_failures == 4 && store.stats().read_failures == 1,
           "failure counters");
    store.release(slot);
    store.release(other);
}

void test_capacity_and_reuse() {
    auto selected           = options("capacity");
    // The segment target is capped at an eighth of the capacity: 16 MiB segments of seven Main
    // slots, nine of which fit in 128 MiB.
    selected.capacity_bytes = 128ULL << 20U;
    ColdContextStore store(selected);
    const auto per_segment = (16ULL << 20U) / kMain;
    const auto segments    = (128ULL << 20U) / (per_segment * kMain);
    std::vector<ColdSlot> slots;
    const auto main_objects = [](std::uint64_t count) {
        std::array<std::uint64_t, kColdObjectClassCount> objects{};
        objects[static_cast<std::size_t>(ColdObjectClass::MainKVPage)] = count;
        return objects;
    };
    expect(store.can_allocate(main_objects(segments * per_segment)) &&
               !store.can_allocate(main_objects(segments * per_segment + 1)),
           "capacity quote disagrees with the segment capacity");
    while (auto slot = store.allocate(ColdObjectClass::MainKVPage)) { slots.push_back(*slot); }
    expect(slots.size() == segments * per_segment, "allocation stops at the segment capacity");
    expect(store.can_allocate(main_objects(0)) && !store.can_allocate(main_objects(1)),
           "capacity quote of a full store accepted another object");
    expect(!store.allocate(ColdObjectClass::StateImage), "other classes share the full capacity");
    expect(segment_files(selected.directory) == segments, "one file per segment");
    // Free one whole segment; its file disappears and the table entry gets a new generation.
    std::set<std::uint32_t> first_segment;
    for (const auto slot : slots) {
        if (slot.segment == slots.front().segment) { store.release(slot); }
    }
    expect(segment_files(selected.directory) == segments - 1, "empty segment unlinked");
    // Every remaining segment is full, so the next slot opens a new file in the freed entry.
    auto reused = store.allocate(ColdObjectClass::MainKVPage);
    expect(reused && reused->segment == slots.front().segment &&
               reused->generation != slots.front().generation,
           "segment entry reused with a new generation");
    expect(!store.valid(slots.front()), "old generation slot rejected");
    store.release(*reused);
    for (const auto slot : slots) {
        if (slot.segment != slots.front().segment) { store.release(slot); }
    }
    expect(segment_files(selected.directory) == 0, "all segments unlinked");
}

void test_backpressure_cancel_and_order() {
    std::atomic<std::uint64_t> slow{0};
    auto selected                     = options("queue");
    selected.io_threads               = 2;
    selected.max_inflight_write_bytes = kState;
    selected.fault_hook               = [&](ColdIoKind, std::uint64_t ticket) {
        if (ticket == slow.load()) { std::this_thread::sleep_for(150ms); }
        return 0;
    };
    ColdContextStore store(selected);
    const auto state = pattern(kState, 5);
    const auto page  = pattern(kMain, 6);
    auto a           = *store.allocate(ColdObjectClass::StateImage);
    auto b           = *store.allocate(ColdObjectClass::MainKVPage);
    slow             = 10;
    expect(store.write(a, state.data(), 10) == ColdSubmit::Submitted, "slow write submitted");
    expect(store.write(b, page.data(), 11) == ColdSubmit::Busy,
           "write beyond the in-flight byte limit is refused");
    bool released = false;
    try {
        store.release(a);
    } catch (const std::logic_error&) { released = true; }
    expect(released, "busy slot cannot be released");
    auto done = drain(store, 1);
    expect(done[0].status == ColdIoStatus::Ok, "slow write completed");
    // Completions are per object and need not follow submission order.
    std::vector<std::byte> out_a(kState), out_b(kMain);
    expect(store.write(b, page.data(), 12) == ColdSubmit::Submitted, "page write");
    auto page_written = drain(store, 1);
    slow              = 13;
    expect(store.read(b, out_b.data(), page_written[0].checksum, 13) == ColdSubmit::Submitted &&
               store.read(a, out_a.data(), done[0].checksum, 14) == ColdSubmit::Submitted,
           "two reads submitted");
    auto order = drain(store, 2);
    expect(order[0].ticket == 14 && order[1].ticket == 13 &&
               order[0].status == ColdIoStatus::Ok && order[1].status == ColdIoStatus::Ok,
           "a later fast read completes before an earlier slow one");
    expect(std::memcmp(out_a.data(), state.data(), kState) == 0 &&
               std::memcmp(out_b.data(), page.data(), kMain) == 0,
           "out-of-order reads return their own bytes");
    // Cancellation of a multi-chunk object with one I/O thread busy on its first chunk.
    auto single           = options("cancel");
    single.io_threads     = 1;
    std::atomic<bool> gate{true};
    single.fault_hook     = [&](ColdIoKind, std::uint64_t) {
        if (gate.exchange(false)) { std::this_thread::sleep_for(100ms); }
        return 0;
    };
    ColdContextStore serial(single);
    auto c = *serial.allocate(ColdObjectClass::StateImage);
    expect(serial.write(c, state.data(), 20) == ColdSubmit::Submitted, "cancel fixture");
    std::this_thread::sleep_for(20ms);
    expect(serial.cancel(20), "cancel accepted while running");
    auto cancelled = drain(serial, 1);
    expect(cancelled[0].status == ColdIoStatus::Cancelled, "unstarted chunks were skipped");
    serial.release(c);
    store.release(a);
    store.release(b);
}

void test_failures_disable() {
    auto selected       = options("faults");
    selected.fault_hook = [](ColdIoKind kind, std::uint64_t) {
        return kind == ColdIoKind::Write ? EIO : 0;
    };
    ColdContextStore store(selected);
    const auto data = pattern(kBackend, 9);
    std::vector<ColdSlot> slots;
    for (std::uint64_t ticket = 0; ticket < 3; ++ticket) {
        slots.push_back(*store.allocate(ColdObjectClass::BackendKVPage));
        expect(store.write(slots.back(), data.data(), ticket) == ColdSubmit::Submitted,
               "failing write submitted");
        auto done = drain(store, 1);
        expect(done[0].status == ColdIoStatus::IoError && done[0].error == EIO,
               "injected write failure is reported");
    }
    expect(store.disabled() && store.stats().write_failures == 3,
           "consecutive failures disable the tier");
    expect(store.stats().write_bytes_24h == store.stats().write_bytes,
           "failed writes reconcile between the 24 h window and the lifetime total");
    expect(!store.allocate(ColdObjectClass::BackendKVPage), "disabled tier allocates nothing");
    expect(store.write(slots.front(), data.data(), 9) == ColdSubmit::Disabled,
           "disabled tier rejects I/O");
    for (const auto slot : slots) { store.release(slot); }
}

void test_timeout_disables() {
    auto selected       = options("timeout");
    selected.io_timeout = 30ms;
    selected.fault_hook = [](ColdIoKind, std::uint64_t) {
        std::this_thread::sleep_for(200ms);
        return 0;
    };
    ColdContextStore store(selected);
    const auto data = pattern(kBackend, 2);
    auto slot       = *store.allocate(ColdObjectClass::BackendKVPage);
    expect(store.write(slot, data.data(), 1) == ColdSubmit::Submitted, "slow write");
    std::this_thread::sleep_for(80ms);
    std::vector<ColdIoCompletion> none;
    store.poll(none);
    expect(none.empty() && store.disabled(), "stuck I/O disables without blocking the caller");
    auto done = drain(store, 1);
    expect(done[0].status == ColdIoStatus::Ok, "late completion is still delivered");
    store.release(slot);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            g_root = argv[1];
        } else if (const char* root = std::getenv("NINFER_TEST_COLD_DIR"); root && *root) {
            g_root = root;
        } else {
            g_root = std::filesystem::current_path() / "cold-store-test";
        }
        // Only this run's own subdirectory is created and removed under a caller-given root.
        g_root /= "store-test-" + std::to_string(static_cast<long>(getpid()));
        std::filesystem::create_directories(g_root);
        test_crc32c();
        test_rejects_tmpfs();
        test_round_trip_and_probe();
        test_corruption_and_lost_file();
        test_capacity_and_reuse();
        test_backpressure_cancel_and_order();
        test_failures_disable();
        test_timeout_disables();
        std::filesystem::remove_all(g_root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures) { return 1; }
    std::cout << "OK cold context store\n";
    return 0;
}
