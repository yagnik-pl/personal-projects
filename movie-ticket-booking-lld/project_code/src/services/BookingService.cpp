#include "BookingService.h"
#include "state/BookingState.h"
#include "state/CreatedState.h"
#include "state/SeatsLockedState.h"
#include "state/CancelledState.h"
#include "../models/Enums.h"
#include <unordered_set>
#include <algorithm>

BookingService::BookingService(
    std::shared_ptr<IBookingRepository> bookingRepo,
    std::shared_ptr<IShowSeatRepository> showSeatRepo,
    std::shared_ptr<IShowRepository> showRepo,
    std::shared_ptr<IUserRepository> userRepo,
    int defaultLockDurationSeconds
) : bookingRepo_(bookingRepo),
    showSeatRepo_(showSeatRepo),
    showRepo_(showRepo),
    userRepo_(userRepo),
    defaultLockDurationSeconds_(defaultLockDurationSeconds) {}

BookingResult BookingService::createBooking(int userId, int showId, const std::vector<int>& seatIds, int lockDurationSeconds) {
    if (userId <= 0) {
        return {false, std::nullopt, "", 0.0, "Invalid user_id: user_id must be a positive integer", 400};
    }

    if (showId <= 0) {
        return {false, std::nullopt, "", 0.0, "Invalid show_id: show_id must be a positive integer", 400};
    }

    if (seatIds.empty()) {
        return {false, std::nullopt, "", 0.0, "seat_ids must not be empty", 400};
    }

    std::unordered_set<int> uniqueSeats;
    for (int seatId : seatIds) {
        if (seatId <= 0) {
            return {false, std::nullopt, "", 0.0, "Invalid seat_id: seat_id must be a positive integer", 400};
        }
        if (!uniqueSeats.insert(seatId).second) {
            return {false, std::nullopt, "", 0.0, "Duplicate seat_ids requested", 400};
        }
    }

    if (userRepo_) {
        auto userOpt = userRepo_->getUser(userId);
        if (!userOpt.has_value()) {
            return {false, std::nullopt, "", 0.0, "User not found", 404};
        }
    }

    if (showRepo_) {
        auto showOpt = showRepo_->getShow(showId);
        if (!showOpt.has_value()) {
            return {false, std::nullopt, "", 0.0, "Show not found", 404};
        }
    }

    double totalPrice = static_cast<double>(seatIds.size()) * SEAT_PRICE;

    Booking newBooking;
    newBooking.user_id = userId;
    newBooking.show_id = showId;
    newBooking.status = BookingStatus::CREATED;
    newBooking.amount = totalPrice;

    Booking createdBooking = bookingRepo_->createBooking(newBooking);
    if (!createdBooking.id.has_value()) {
        return {false, std::nullopt, "", 0.0, "Failed to create booking in database", 500};
    }

    int bookingId = *createdBooking.id;
    BookingContext context(bookingId, std::make_shared<CreatedState>());

    int lockDuration = (lockDurationSeconds > 0) ? lockDurationSeconds : defaultLockDurationSeconds_;
    bool lockSuccess = showSeatRepo_->lockSeats(showId, seatIds, bookingId, lockDuration);

    if (!lockSuccess) {
        context.cancel();
        bookingRepo_->updateBookingStatus(bookingId, BookingStatus::CANCELLED);
        return {false, bookingId, BookingStatus::CANCELLED, totalPrice, "One or more requested seats are already locked or booked", 409};
    }

    context.lockSeats();
    bookingRepo_->updateBookingStatus(bookingId, BookingStatus::SEATS_LOCKED);

    return {true, bookingId, BookingStatus::SEATS_LOCKED, totalPrice, "", 201};
}

std::optional<Booking> BookingService::getBooking(int bookingId) {
    if (bookingId <= 0) {
        return std::nullopt;
    }
    return bookingRepo_->getBooking(bookingId);
}

bool BookingService::cancelBooking(int bookingId) {
    if (bookingId <= 0) {
        return false;
    }

    auto bookingOpt = bookingRepo_->getBooking(bookingId);
    if (!bookingOpt.has_value()) {
        return false;
    }

    const auto& booking = *bookingOpt;
    if (booking.status == BookingStatus::CONFIRMED || booking.status == BookingStatus::CANCELLED) {
        return false;
    }

    try {
        BookingContext context(bookingId, booking.status);
        context.cancel();
    } catch (...) {
        return false;
    }

    bool updated = bookingRepo_->updateBookingStatus(bookingId, BookingStatus::CANCELLED);
    showSeatRepo_->releaseSeatsForBooking(bookingId);
    return updated;
}

std::vector<Booking> BookingService::getBookingsByUser(int userId) {
    if (userId <= 0) {
        return {};
    }
    return bookingRepo_->getBookingsByUser(userId);
}

std::vector<Booking> BookingService::getBookingsByShow(int showId) {
    if (showId <= 0) {
        return {};
    }
    return bookingRepo_->getBookingsByShow(showId);
}
