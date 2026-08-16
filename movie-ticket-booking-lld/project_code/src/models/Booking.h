#pragma once
#include <string>
#include <optional>

struct Booking {
    std::optional<int> id;
    int user_id;
    int show_id;
    std::string status; // 'CREATED', 'SEATS_LOCKED', 'PAYMENT_PENDING', 'CONFIRMED', 'CANCELLED'
    double amount;
    std::optional<std::string> created_at;
};
