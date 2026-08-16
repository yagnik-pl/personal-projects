#pragma once
#include "ILockCleanupService.h"
#include "../repositories/IShowSeatRepository.h"
#include "../repositories/IBookingRepository.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

class LockCleanupService : public ILockCleanupService {
public:
    LockCleanupService(
        std::shared_ptr<IShowSeatRepository> showSeatRepo,
        std::shared_ptr<IBookingRepository> bookingRepo,
        std::chrono::milliseconds interval = std::chrono::milliseconds(500)
    );
    ~LockCleanupService() override;

    void start() override;
    void stop() override;
    bool isRunning() const override;
    int cleanupExpiredLocks() override;

    void setInterval(std::chrono::milliseconds interval) { interval_ = interval; }
    std::chrono::milliseconds getInterval() const { return interval_; }

private:
    std::shared_ptr<IShowSeatRepository> showSeatRepo_;
    std::shared_ptr<IBookingRepository> bookingRepo_;
    std::chrono::milliseconds interval_;
    std::atomic<bool> running_{false};
    std::thread workerThread_;
    std::mutex cvMutex_;
    std::condition_variable cv_;
};
