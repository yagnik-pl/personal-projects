#include <gtest/gtest.h>
#include "services/LockCleanupService.h"
#include "repositories/IShowSeatRepository.h"
#include "repositories/IBookingRepository.h"
#include "models/Enums.h"
#include <map>
#include <thread>
#include <chrono>

class FakeBookingRepositoryForCleanup : public IBookingRepository {
public:
    Booking createBooking(const Booking& booking) override {
        Booking b = booking;
        b.id = nextId_++;
        bookings_[*b.id] = b;
        return b;
    }

    std::optional<Booking> getBooking(int id) override {
        auto it = bookings_.find(id);
        if (it != bookings_.end()) return it->second;
        return std::nullopt;
    }

    bool updateBookingStatus(int id, const std::string& status) override {
        auto it = bookings_.find(id);
        if (it != bookings_.end()) {
            it->second.status = status;
            return true;
        }
        return false;
    }

    bool updateBookingAmount(int id, double amount) override { return false; }
    std::vector<Booking> getBookingsByUser(int userId) override { return {}; }
    std::vector<Booking> getBookingsByShow(int showId) override { return {}; }

    void addBooking(const Booking& b) {
        if (b.id.has_value()) {
            bookings_[*b.id] = b;
        }
    }

private:
    int nextId_{1001};
    std::map<int, Booking> bookings_;
};

class FakeShowSeatRepositoryForCleanup : public IShowSeatRepository {
public:
    ShowSeat createShowSeat(const ShowSeat& showSeat) override { return showSeat; }
    std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) override { return showSeats; }
    std::optional<ShowSeat> getShowSeat(int showId, int seatId) override { return std::nullopt; }
    std::vector<ShowSeat> getShowSeatsByShow(int showId) override { return {}; }
    bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) override { return true; }
    bool confirmSeats(int bookingId) override { return true; }
    bool releaseSeatsForBooking(int bookingId) override { return true; }

    std::vector<int> releaseExpiredSeats() override {
        auto result = expiredIdsToReturn_;
        expiredIdsToReturn_.clear();
        cleanupCallCount_++;
        return result;
    }

    void setExpiredBookings(const std::vector<int>& ids) {
        expiredIdsToReturn_ = ids;
    }

    std::vector<int> expiredIdsToReturn_;
    int cleanupCallCount_{0};
};

class LockCleanupServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        showSeatRepo_ = std::make_shared<FakeShowSeatRepositoryForCleanup>();
        bookingRepo_ = std::make_shared<FakeBookingRepositoryForCleanup>();
    }

    std::shared_ptr<FakeShowSeatRepositoryForCleanup> showSeatRepo_;
    std::shared_ptr<FakeBookingRepositoryForCleanup> bookingRepo_;
};

TEST_F(LockCleanupServiceTest, LifecycleStartStop) {
    LockCleanupService service(showSeatRepo_, bookingRepo_, std::chrono::milliseconds(50));
    EXPECT_FALSE(service.isRunning());

    service.start();
    EXPECT_TRUE(service.isRunning());

    // Calling start again is safe (idempotent)
    service.start();
    EXPECT_TRUE(service.isRunning());

    service.stop();
    EXPECT_FALSE(service.isRunning());

    // Calling stop again is safe (idempotent)
    service.stop();
    EXPECT_FALSE(service.isRunning());
}

TEST_F(LockCleanupServiceTest, CleanupExpiredLocksCancelsUnconfirmedBookings) {
    LockCleanupService service(showSeatRepo_, bookingRepo_, std::chrono::milliseconds(50));

    // Add unconfirmed booking
    Booking b1;
    b1.id = 1001;
    b1.user_id = 1;
    b1.show_id = 1;
    b1.status = BookingStatus::SEATS_LOCKED;
    b1.amount = 50.0;
    bookingRepo_->addBooking(b1);

    // Add already confirmed booking
    Booking b2;
    b2.id = 1002;
    b2.user_id = 2;
    b2.show_id = 1;
    b2.status = BookingStatus::CONFIRMED;
    b2.amount = 25.0;
    bookingRepo_->addBooking(b2);

    // Expired locks reported for 1001 and 1002
    showSeatRepo_->setExpiredBookings({1001, 1002});

    int cancelledCount = service.cleanupExpiredLocks();
    EXPECT_EQ(cancelledCount, 1);

    // Booking 1001 should be CANCELLED
    auto b1After = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(b1After.has_value());
    EXPECT_EQ(b1After->status, BookingStatus::CANCELLED);

    // Booking 1002 should remain CONFIRMED
    auto b2After = bookingRepo_->getBooking(1002);
    ASSERT_TRUE(b2After.has_value());
    EXPECT_EQ(b2After->status, BookingStatus::CONFIRMED);
}

TEST_F(LockCleanupServiceTest, BackgroundThreadExecutesPeriodically) {
    LockCleanupService service(showSeatRepo_, bookingRepo_, std::chrono::milliseconds(20));

    Booking b;
    b.id = 2001;
    b.user_id = 1;
    b.show_id = 1;
    b.status = BookingStatus::SEATS_LOCKED;
    b.amount = 25.0;
    bookingRepo_->addBooking(b);

    showSeatRepo_->setExpiredBookings({2001});

    service.start();
    // Wait for background worker to trigger at least once
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    service.stop();

    EXPECT_GT(showSeatRepo_->cleanupCallCount_, 0);

    auto bAfter = bookingRepo_->getBooking(2001);
    ASSERT_TRUE(bAfter.has_value());
    EXPECT_EQ(bAfter->status, BookingStatus::CANCELLED);
}
