#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "services/BookingService.h"
#include "repositories/IBookingRepository.h"
#include "repositories/IShowSeatRepository.h"
#include "repositories/IShowRepository.h"
#include "repositories/IUserRepository.h"
#include "models/Enums.h"
#include <map>

using ::testing::_;
using ::testing::Return;

class FakeBookingRepository : public IBookingRepository {
public:
    Booking createBooking(const Booking& booking) override {
        Booking b = booking;
        b.id = nextId_++;
        bookings_[*b.id] = b;
        return b;
    }

    std::optional<Booking> getBooking(int id) override {
        auto it = bookings_.find(id);
        if (it != bookings_.end()) {
            return it->second;
        }
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

    bool updateBookingAmount(int id, double amount) override {
        auto it = bookings_.find(id);
        if (it != bookings_.end()) {
            it->second.amount = amount;
            return true;
        }
        return false;
    }

    std::vector<Booking> getBookingsByUser(int userId) override {
        std::vector<Booking> result;
        for (const auto& [id, b] : bookings_) {
            if (b.user_id == userId) result.push_back(b);
        }
        return result;
    }

    std::vector<Booking> getBookingsByShow(int showId) override {
        std::vector<Booking> result;
        for (const auto& [id, b] : bookings_) {
            if (b.show_id == showId) result.push_back(b);
        }
        return result;
    }

private:
    int nextId_{1001};
    std::map<int, Booking> bookings_;
};

class FakeShowSeatRepository : public IShowSeatRepository {
public:
    bool shouldFailLock{false};

    ShowSeat createShowSeat(const ShowSeat& showSeat) override { return showSeat; }
    std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) override { return showSeats; }
    std::optional<ShowSeat> getShowSeat(int showId, int seatId) override { return std::nullopt; }
    std::vector<ShowSeat> getShowSeatsByShow(int showId) override { return {}; }

    bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) override {
        if (shouldFailLock) return false;
        lastLockedSeats_ = seatIds;
        lastBookingId_ = bookingId;
        return true;
    }

    bool confirmSeats(int bookingId) override { return true; }
    bool releaseSeatsForBooking(int bookingId) override {
        releasedBookings_.push_back(bookingId);
        return true;
    }
    std::vector<int> releaseExpiredSeats() override { return {}; }

    std::vector<int> lastLockedSeats_;
    int lastBookingId_{0};
    std::vector<int> releasedBookings_;
};

class FakeUserRepository : public IUserRepository {
public:
    User createUser(const User& user) override { return user; }
    std::optional<User> getUser(int id) override {
        if (id == 999) return std::nullopt; // Simulates non-existent user
        User u;
        u.id = id;
        u.name = "Test User";
        u.email = "test@example.com";
        return u;
    }
    std::optional<User> getUserByEmail(const std::string& email) override { return std::nullopt; }
    std::vector<User> getAllUsers() override { return {}; }
};

class FakeShowRepository : public IShowRepository {
public:
    Show createShow(const Show& show) override { return show; }
    std::optional<Show> getShow(int id) override {
        if (id == 999) return std::nullopt; // Simulates non-existent show
        Show s;
        s.id = id;
        s.movie_id = 1;
        s.screen_id = 1;
        return s;
    }
    std::vector<Show> getAllShows() override { return {}; }
    std::vector<Show> getShowsByMovie(int movieId) override { return {}; }
    std::vector<Show> getShowsByScreen(int screenId) override { return {}; }
};

class BookingServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        bookingRepo_ = std::make_shared<FakeBookingRepository>();
        showSeatRepo_ = std::make_shared<FakeShowSeatRepository>();
        showRepo_ = std::make_shared<FakeShowRepository>();
        userRepo_ = std::make_shared<FakeUserRepository>();

        service_ = std::make_unique<BookingService>(
            bookingRepo_, showSeatRepo_, showRepo_, userRepo_, 300
        );
    }

    std::shared_ptr<FakeBookingRepository> bookingRepo_;
    std::shared_ptr<FakeShowSeatRepository> showSeatRepo_;
    std::shared_ptr<FakeShowRepository> showRepo_;
    std::shared_ptr<FakeUserRepository> userRepo_;
    std::unique_ptr<BookingService> service_;
};

TEST_F(BookingServiceTest, SuccessfulBookingCreation) {
    auto result = service_->createBooking(1, 1, {10, 11});

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.httpStatusCode, 201);
    EXPECT_TRUE(result.bookingId.has_value());
    EXPECT_EQ(result.status, BookingStatus::SEATS_LOCKED);
    EXPECT_DOUBLE_EQ(result.totalPrice, 50.0); // 2 seats * 25.0

    auto booking = service_->getBooking(*result.bookingId);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::SEATS_LOCKED);
    EXPECT_DOUBLE_EQ(booking->amount, 50.0);
}

TEST_F(BookingServiceTest, SingleSeatBookingPrice) {
    auto result = service_->createBooking(1, 1, {5});

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.httpStatusCode, 201);
    EXPECT_DOUBLE_EQ(result.totalPrice, 25.0); // 1 seat * 25.0
}

TEST_F(BookingServiceTest, LockFailureTransitionsToCancelled) {
    showSeatRepo_->shouldFailLock = true;
    auto result = service_->createBooking(1, 1, {10});

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 409);
    EXPECT_EQ(result.status, BookingStatus::CANCELLED);
    ASSERT_TRUE(result.bookingId.has_value());

    auto booking = service_->getBooking(*result.bookingId);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::CANCELLED);
}

TEST_F(BookingServiceTest, InvalidInputs) {
    // Empty seat list
    auto r1 = service_->createBooking(1, 1, {});
    EXPECT_FALSE(r1.success);
    EXPECT_EQ(r1.httpStatusCode, 400);

    // Invalid user ID
    auto r2 = service_->createBooking(-1, 1, {1});
    EXPECT_FALSE(r2.success);
    EXPECT_EQ(r2.httpStatusCode, 400);

    // Invalid show ID
    auto r3 = service_->createBooking(1, 0, {1});
    EXPECT_FALSE(r3.success);
    EXPECT_EQ(r3.httpStatusCode, 400);

    // Duplicate seat IDs
    auto r4 = service_->createBooking(1, 1, {2, 2});
    EXPECT_FALSE(r4.success);
    EXPECT_EQ(r4.httpStatusCode, 400);

    // Negative seat ID
    auto r5 = service_->createBooking(1, 1, {-5});
    EXPECT_FALSE(r5.success);
    EXPECT_EQ(r5.httpStatusCode, 400);

    // Non-existent user
    auto r6 = service_->createBooking(999, 1, {1});
    EXPECT_FALSE(r6.success);
    EXPECT_EQ(r6.httpStatusCode, 404);

    // Non-existent show
    auto r7 = service_->createBooking(1, 999, {1});
    EXPECT_FALSE(r7.success);
    EXPECT_EQ(r7.httpStatusCode, 404);
}

TEST_F(BookingServiceTest, CancelBooking) {
    auto result = service_->createBooking(1, 1, {1, 2});
    ASSERT_TRUE(result.success);
    int bookingId = *result.bookingId;

    EXPECT_TRUE(service_->cancelBooking(bookingId));

    auto booking = service_->getBooking(bookingId);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::CANCELLED);

    // Cancelling again should return false
    EXPECT_FALSE(service_->cancelBooking(bookingId));

    // Invalid booking ID
    EXPECT_FALSE(service_->cancelBooking(-1));
    EXPECT_FALSE(service_->cancelBooking(99999));
}
