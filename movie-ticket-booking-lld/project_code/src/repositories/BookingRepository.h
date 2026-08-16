#pragma once
#include "IBookingRepository.h"
#include "../core/Database.h"
#include <memory>

class BookingRepository : public IBookingRepository {
public:
    explicit BookingRepository(std::shared_ptr<Database> db);
    Booking createBooking(const Booking& booking) override;
    std::optional<Booking> getBooking(int id) override;
    bool updateBookingStatus(int id, const std::string& status) override;
    bool updateBookingAmount(int id, double amount) override;
    std::vector<Booking> getBookingsByUser(int userId) override;
    std::vector<Booking> getBookingsByShow(int showId) override;

private:
    std::shared_ptr<Database> db_;
};
