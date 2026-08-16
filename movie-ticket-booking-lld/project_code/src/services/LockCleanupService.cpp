#include "LockCleanupService.h"
#include "../models/Enums.h"

LockCleanupService::LockCleanupService(
    std::shared_ptr<IShowSeatRepository> showSeatRepo,
    std::shared_ptr<IBookingRepository> bookingRepo,
    std::chrono::milliseconds interval
) : showSeatRepo_(showSeatRepo),
    bookingRepo_(bookingRepo),
    interval_(interval) {}

LockCleanupService::~LockCleanupService() {
    stop();
}

void LockCleanupService::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return; // Already running
    }

    workerThread_ = std::thread([this]() {
        while (running_.load()) {
            {
                std::unique_lock<std::mutex> lock(cvMutex_);
                cv_.wait_for(lock, interval_, [this]() { return !running_.load(); });
            }
            if (!running_.load()) {
                break;
            }
            try {
                cleanupExpiredLocks();
            } catch (...) {
                // Ignore errors in background loop to ensure reaper keeps running
            }
        }
    });
}

void LockCleanupService::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return; // Already stopped
    }

    {
        std::lock_guard<std::mutex> lock(cvMutex_);
        cv_.notify_all();
    }

    if (workerThread_.joinable()) {
        workerThread_.join();
    }
}

bool LockCleanupService::isRunning() const {
    return running_.load();
}

int LockCleanupService::cleanupExpiredLocks() {
    if (!showSeatRepo_ || !bookingRepo_) {
        return 0;
    }

    std::vector<int> expiredBookingIds = showSeatRepo_->releaseExpiredSeats();
    int cancelledCount = 0;

    for (int bookingId : expiredBookingIds) {
        if (bookingId <= 0) {
            continue;
        }

        auto bookingOpt = bookingRepo_->getBooking(bookingId);
        if (bookingOpt.has_value()) {
            const auto& booking = *bookingOpt;
            if (booking.status == BookingStatus::CREATED ||
                booking.status == BookingStatus::SEATS_LOCKED ||
                booking.status == BookingStatus::PAYMENT_PENDING) {
                if (bookingRepo_->updateBookingStatus(bookingId, BookingStatus::CANCELLED)) {
                    cancelledCount++;
                }
            }
        }
    }

    return cancelledCount;
}
