#pragma once
#include <optional>
#include <vector>
#include <string>
#include "../models/Booking.h"

class IBookingRepository {
public:
    virtual ~IBookingRepository() = default;
    virtual Booking createBooking(const Booking& booking) = 0;
    virtual std::optional<Booking> getBooking(int id) = 0;
    virtual bool updateBookingStatus(int id, const std::string& status) = 0;
    virtual bool updateBookingAmount(int id, double amount) = 0;
    virtual std::vector<Booking> getBookingsByUser(int userId) = 0;
    virtual std::vector<Booking> getBookingsByShow(int showId) = 0;
};
