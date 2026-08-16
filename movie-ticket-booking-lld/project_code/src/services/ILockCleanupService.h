#pragma once
#include <chrono>

class ILockCleanupService {
public:
    virtual ~ILockCleanupService() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual int cleanupExpiredLocks() = 0;
};
