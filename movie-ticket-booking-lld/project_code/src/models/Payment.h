#pragma once
#include <string>
#include <optional>

struct Payment {
    std::optional<int> id;
    int booking_id;
    std::string status; // 'PENDING', 'SUCCESS', 'FAILED'
    std::optional<std::string> transaction_id;
    std::optional<std::string> created_at;
};
