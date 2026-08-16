#pragma once
#include "IBookingService.h"
#include "../repositories/IBookingRepository.h"
#include "../repositories/IShowSeatRepository.h"
#include "../repositories/IShowRepository.h"
#include "../repositories/IUserRepository.h"
#include <memory>

class BookingService : public IBookingService {
public:
    static constexpr double SEAT_PRICE = 25.0;

    BookingService(
        std::shared_ptr<IBookingRepository> bookingRepo,
        std::shared_ptr<IShowSeatRepository> showSeatRepo,
        std::shared_ptr<IShowRepository> showRepo = nullptr,
        std::shared_ptr<IUserRepository> userRepo = nullptr,
        int defaultLockDurationSeconds = 300
    );

    BookingResult createBooking(int userId, int showId, const std::vector<int>& seatIds, int lockDurationSeconds = 300) override;
    std::optional<Booking> getBooking(int bookingId) override;
    bool cancelBooking(int bookingId) override;
    std::vector<Booking> getBookingsByUser(int userId) override;
    std::vector<Booking> getBookingsByShow(int showId) override;

    void setDefaultLockDuration(int seconds) { defaultLockDurationSeconds_ = seconds; }
    int getDefaultLockDuration() const { return defaultLockDurationSeconds_; }

private:
    std::shared_ptr<IBookingRepository> bookingRepo_;
    std::shared_ptr<IShowSeatRepository> showSeatRepo_;
    std::shared_ptr<IShowRepository> showRepo_;
    std::shared_ptr<IUserRepository> userRepo_;
    int defaultLockDurationSeconds_;
};
