#pragma once
#include <string>
#include <optional>

struct ShowSeat {
    std::optional<int> id;
    int show_id;
    int seat_id;
    std::string status; // 'AVAILABLE', 'LOCKED', 'BOOKED'
    std::optional<std::string> lock_expiry_time;
    std::optional<int> booking_id;
};
