#pragma once
#include <string>

namespace SeatStatus {
    inline const std::string AVAILABLE = "AVAILABLE";
    inline const std::string LOCKED = "LOCKED";
    inline const std::string BOOKED = "BOOKED";
}

namespace BookingStatus {
    inline const std::string CREATED = "CREATED";
    inline const std::string SEATS_LOCKED = "SEATS_LOCKED";
    inline const std::string PAYMENT_PENDING = "PAYMENT_PENDING";
    inline const std::string CONFIRMED = "CONFIRMED";
    inline const std::string CANCELLED = "CANCELLED";
}

namespace PaymentStatus {
    inline const std::string PENDING = "PENDING";
    inline const std::string SUCCESS = "SUCCESS";
    inline const std::string FAILED = "FAILED";
}
