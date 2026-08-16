#include <gtest/gtest.h>
#include "services/PaymentService.h"
#include "services/payment/MockPaymentGateway.h"
#include "repositories/IPaymentRepository.h"
#include "repositories/IBookingRepository.h"
#include "repositories/IShowSeatRepository.h"
#include "models/Enums.h"
#include <map>

class FakePaymentRepository : public IPaymentRepository {
public:
    Payment createPayment(const Payment& payment) override {
        Payment p = payment;
        p.id = nextId_++;
        payments_[*p.id] = p;
        bookingPayments_[p.booking_id] = p;
        return p;
    }

    std::optional<Payment> getPayment(int id) override {
        auto it = payments_.find(id);
        if (it != payments_.end()) return it->second;
        return std::nullopt;
    }

    std::optional<Payment> getPaymentByBookingId(int bookingId) override {
        auto it = bookingPayments_.find(bookingId);
        if (it != bookingPayments_.end()) return it->second;
        return std::nullopt;
    }

    bool updatePaymentStatus(int id, const std::string& status, const std::optional<std::string>& transactionId) override {
        auto it = payments_.find(id);
        if (it != payments_.end()) {
            it->second.status = status;
            it->second.transaction_id = transactionId;
            return true;
        }
        return false;
    }

private:
    int nextId_{5001};
    std::map<int, Payment> payments_;
    std::map<int, Payment> bookingPayments_;
};

class FakeBookingRepositoryForPayment : public IBookingRepository {
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

class FakeShowSeatRepositoryForPayment : public IShowSeatRepository {
public:
    ShowSeat createShowSeat(const ShowSeat& showSeat) override { return showSeat; }
    std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) override { return showSeats; }
    std::optional<ShowSeat> getShowSeat(int showId, int seatId) override { return std::nullopt; }
    std::vector<ShowSeat> getShowSeatsByShow(int showId) override { return {}; }
    bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) override { return true; }

    bool confirmSeats(int bookingId) override {
        confirmedBookings_.push_back(bookingId);
        return true;
    }

    bool releaseSeatsForBooking(int bookingId) override { return true; }
    std::vector<int> releaseExpiredSeats() override { return {}; }

    std::vector<int> confirmedBookings_;
};

class PaymentServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        paymentRepo_ = std::make_shared<FakePaymentRepository>();
        bookingRepo_ = std::make_shared<FakeBookingRepositoryForPayment>();
        showSeatRepo_ = std::make_shared<FakeShowSeatRepositoryForPayment>();
        gateway_ = std::make_shared<MockPaymentGateway>();

        service_ = std::make_unique<PaymentService>(
            paymentRepo_, bookingRepo_, showSeatRepo_, gateway_
        );

        // Set up standard booking 1001 (SEATS_LOCKED, amount 50.0)
        Booking b;
        b.id = 1001;
        b.user_id = 1;
        b.show_id = 1;
        b.status = BookingStatus::SEATS_LOCKED;
        b.amount = 50.0;
        bookingRepo_->addBooking(b);
    }

    std::shared_ptr<FakePaymentRepository> paymentRepo_;
    std::shared_ptr<FakeBookingRepositoryForPayment> bookingRepo_;
    std::shared_ptr<FakeShowSeatRepositoryForPayment> showSeatRepo_;
    std::shared_ptr<MockPaymentGateway> gateway_;
    std::unique_ptr<PaymentService> service_;
};

TEST_F(PaymentServiceTest, SuccessfulPayment) {
    auto result = service_->processPayment(1001, 50.0, false);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.httpStatusCode, 200);
    EXPECT_EQ(result.status, PaymentStatus::SUCCESS);
    ASSERT_TRUE(result.paymentId.has_value());
    ASSERT_TRUE(result.transactionId.has_value());
    EXPECT_FALSE(result.transactionId->empty());

    // Check booking transitioned to CONFIRMED
    auto booking = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::CONFIRMED);

    // Check seats confirmed
    EXPECT_EQ(showSeatRepo_->confirmedBookings_.size(), 1);
    EXPECT_EQ(showSeatRepo_->confirmedBookings_[0], 1001);

    // Check payment record
    auto payment = service_->getPayment(*result.paymentId);
    ASSERT_TRUE(payment.has_value());
    EXPECT_EQ(payment->status, PaymentStatus::SUCCESS);
}

TEST_F(PaymentServiceTest, MockPaymentFailure) {
    auto result = service_->processPayment(1001, 50.0, true);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 400);
    EXPECT_EQ(result.status, PaymentStatus::FAILED);
    EXPECT_EQ(result.errorMessage, "Mock payment failure requested");

    // Booking remains in SEATS_LOCKED
    auto booking = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::SEATS_LOCKED);

    // No seats confirmed
    EXPECT_EQ(showSeatRepo_->confirmedBookings_.size(), 0);
}

TEST_F(PaymentServiceTest, PaymentFailureThenRetrySuccess) {
    // 1st attempt: fails
    auto failResult = service_->processPayment(1001, 50.0, true);
    EXPECT_FALSE(failResult.success);
    EXPECT_EQ(failResult.httpStatusCode, 400);

    auto bookingMid = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(bookingMid.has_value());
    EXPECT_NE(bookingMid->status, BookingStatus::CONFIRMED);

    // 2nd attempt: succeeds
    auto successResult = service_->processPayment(1001, 50.0, false);
    EXPECT_TRUE(successResult.success);
    EXPECT_EQ(successResult.httpStatusCode, 200);

    auto bookingFinal = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(bookingFinal.has_value());
    EXPECT_EQ(bookingFinal->status, BookingStatus::CONFIRMED);
}

TEST_F(PaymentServiceTest, AmountMismatchFails) {
    auto result = service_->processPayment(1001, 25.0, false);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 400);

    auto booking = bookingRepo_->getBooking(1001);
    ASSERT_TRUE(booking.has_value());
    EXPECT_EQ(booking->status, BookingStatus::SEATS_LOCKED);
}

TEST_F(PaymentServiceTest, NonExistentBookingFails) {
    auto result = service_->processPayment(999999, 50.0, false);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 404);
}

TEST_F(PaymentServiceTest, InvalidNegativeAmountFails) {
    auto result = service_->processPayment(1001, -50.0, false);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 400);
}

TEST_F(PaymentServiceTest, AlreadyConfirmedBookingFails) {
    // Confirm first
    auto res1 = service_->processPayment(1001, 50.0, false);
    EXPECT_TRUE(res1.success);

    // Attempting payment again should conflict
    auto res2 = service_->processPayment(1001, 50.0, false);
    EXPECT_FALSE(res2.success);
    EXPECT_EQ(res2.httpStatusCode, 409);
}

TEST_F(PaymentServiceTest, CancelledBookingFails) {
    Booking b;
    b.id = 1002;
    b.user_id = 2;
    b.show_id = 1;
    b.status = BookingStatus::CANCELLED;
    b.amount = 25.0;
    bookingRepo_->addBooking(b);

    auto result = service_->processPayment(1002, 25.0, false);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.httpStatusCode, 409);
}
