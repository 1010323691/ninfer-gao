#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer {

// CRC-32C (Castagnoli, reflected, init/xorout 0xFFFFFFFF) using the SSE4.2 instruction.
// Callers check crc32c_supported() once; the functions below require it.
[[nodiscard]] bool crc32c_supported() noexcept;

[[nodiscard]] std::uint32_t crc32c(const std::byte* data, std::size_t bytes) noexcept;

// Independent CRC-32C of each consecutive `block_bytes` block of [data, data + bytes); the last
// block may be shorter. Up to four full blocks are computed in interleaved streams, so the result
// is identical to calling crc32c() per block while using more of the instruction throughput.
void crc32c_blocks(const std::byte* data, std::size_t bytes, std::size_t block_bytes,
                   std::span<std::uint32_t> out) noexcept;

} // namespace ninfer
