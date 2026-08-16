#pragma once
#include <optional>
#include <vector>
#include <string>
#include "../models/Booking.h"

struct BookingResult {
    bool success{false};
    std::optional<int> bookingId;
    std::string status;
    double totalPrice{0.0};
    std::string errorMessage;
    int httpStatusCode{200};
};

class IBookingService {
public:
    virtual ~IBookingService() = default;

    virtual BookingResult createBooking(int userId, int showId, const std::vector<int>& seatIds, int lockDurationSeconds = 300) = 0;
    virtual std::optional<Booking> getBooking(int bookingId) = 0;
    virtual bool cancelBooking(int bookingId) = 0;
    virtual std::vector<Booking> getBookingsByUser(int userId) = 0;
    virtual std::vector<Booking> getBookingsByShow(int showId) = 0;
};
