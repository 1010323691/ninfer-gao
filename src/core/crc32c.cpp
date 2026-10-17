#include "core/crc32c.h"

#include <cstring>
#include <nmmintrin.h>

namespace ninfer {
namespace {

__attribute__((target("sse4.2"))) std::uint32_t update(std::uint32_t crc, const std::byte* data,
                                                       std::size_t bytes) noexcept {
    std::uint64_t state = crc;
    while (bytes >= 8) {
        std::uint64_t word;
        std::memcpy(&word, data, sizeof(word));
        state = _mm_crc32_u64(state, word);
        data += 8;
        bytes -= 8;
    }
    auto narrow = static_cast<std::uint32_t>(state);
    while (bytes != 0) {
        narrow = _mm_crc32_u8(narrow, static_cast<unsigned char>(*data));
        ++data;
        --bytes;
    }
    return narrow;
}

// Four independent streams hide the three-cycle CRC32 latency behind one-per-cycle throughput.
__attribute__((target("sse4.2"))) void update4(const std::byte* data, std::size_t block_bytes,
                                               std::uint32_t* out) noexcept {
    std::uint64_t a = 0xFFFFFFFFU, b = 0xFFFFFFFFU, c = 0xFFFFFFFFU, d = 0xFFFFFFFFU;
    const std::byte* pa = data;
    const std::byte* pb = data + block_bytes;
    const std::byte* pc = data + 2 * block_bytes;
    const std::byte* pd = data + 3 * block_bytes;
    std::size_t words   = block_bytes / 8;
    for (std::size_t index = 0; index < words; ++index) {
        std::uint64_t wa, wb, wc, wd;
        std::memcpy(&wa, pa + index * 8, 8);
        std::memcpy(&wb, pb + index * 8, 8);
        std::memcpy(&wc, pc + index * 8, 8);
        std::memcpy(&wd, pd + index * 8, 8);
        a = _mm_crc32_u64(a, wa);
        b = _mm_crc32_u64(b, wb);
        c = _mm_crc32_u64(c, wc);
        d = _mm_crc32_u64(d, wd);
    }
    const std::size_t tail = words * 8;
    out[0] = ~update(static_cast<std::uint32_t>(a), pa + tail, block_bytes - tail);
    out[1] = ~update(static_cast<std::uint32_t>(b), pb + tail, block_bytes - tail);
    out[2] = ~update(static_cast<std::uint32_t>(c), pc + tail, block_bytes - tail);
    out[3] = ~update(static_cast<std::uint32_t>(d), pd + tail, block_bytes - tail);
}

} // namespace

bool crc32c_supported() noexcept { return __builtin_cpu_supports("sse4.2"); }

std::uint32_t crc32c(const std::byte* data, std::size_t bytes) noexcept {
    return ~update(0xFFFFFFFFU, data, bytes);
}

void crc32c_blocks(const std::byte* data, std::size_t bytes, std::size_t block_bytes,
                   std::span<std::uint32_t> out) noexcept {
    std::size_t index = 0;
    while (bytes >= 4 * block_bytes && index + 4 <= out.size()) {
        update4(data, block_bytes, out.data() + index);
        data += 4 * block_bytes;
        bytes -= 4 * block_bytes;
        index += 4;
    }
    while (bytes != 0 && index < out.size()) {
        const std::size_t length = bytes < block_bytes ? bytes : block_bytes;
        out[index++]             = crc32c(data, length);
        data += length;
        bytes -= length;
    }
}

} // namespace ninfer
