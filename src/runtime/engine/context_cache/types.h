#pragma once

#include <cstdint>

namespace ninfer::runtime {

using ContinuationOwnerToken = std::uint64_t;

struct CacheRetentionPriority {
    bool reused               = false;
    std::uint64_t last_demand = 0;
};

// Spilling: Host-pressure victims are being written to the SSD tier; their Host bytes are
// released when that write completes, so the caller waits like for a context transaction.
enum class ReclaimProgress : std::uint8_t { Blocked, Changed, Transferring, Spilling };

enum class ReclaimPurpose : std::uint8_t { Execution, FreshAdmission, OptionalWrite };

struct ReclaimRights {
    ReclaimPurpose purpose = ReclaimPurpose::Execution;
    std::uint64_t request  = 0;
};

} // namespace ninfer::runtime
