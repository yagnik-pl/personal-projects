#include <gtest/gtest.h>
#include <crow.h>
#include <memory>
#include <vector>
#include <optional>
#include <string>

#include "controllers/ShowController.h"
#include "controllers/BookingController.h"
#include "controllers/PaymentController.h"
#include "controllers/ResetController.h"
#include "services/IShowService.h"
#include "services/IBookingService.h"
#include "services/IPaymentService.h"
#include "models/Enums.h"

// --- Mock Services for Controller Testing ---

class MockShowService : public IShowService {
public:
    std::vector<Show> getAllShows() override {
        return mockShows;
    }

    std::optional<Show> getShow(int showId) override {
        for (const auto& s : mockShows) {
            if (s.id.has_value() && *s.id == showId) {
                return s;
            }
        }
        return std::nullopt;
    }

    std::vector<ShowSeat> getShowSeats(int showId) override {
        if (showId == 1) {
            return mockSeats;
        }
        return {};
    }

    std::vector<Show> getShowsByMovie(int movieId) override {
        return {};
    }

    std::vector<Show> mockShows;
    std::vector<ShowSeat> mockSeats;
};

class MockBookingService : public IBookingService {
public:
    BookingResult createBooking(int userId, int showId, const std::vector<int>& seatIds, int lockDurationSeconds = 300) override {
        if (userId <= 0 || showId <= 0 || seatIds.empty()) {
            return {false, std::nullopt, "", 0.0, "Invalid arguments", 400};
        }
        if (showId == 999) {
            return {false, std::nullopt, "", 0.0, "Show not found", 404};
        }
        if (seatIds.size() == 1 && seatIds[0] == 99) {
            return {false, std::nullopt, "", 0.0, "One or more requested seats are already locked or booked", 409};
        }
        return {true, 1001, BookingStatus::SEATS_LOCKED, static_cast<double>(seatIds.size()) * 25.0, "", 201};
    }

    std::optional<Booking> getBooking(int bookingId) override {
        if (bookingId == 1001) {
            Booking b;
            b.id = 1001;
            b.user_id = 1;
            b.show_id = 1;
            b.status = BookingStatus::SEATS_LOCKED;
            b.amount = 50.0;
            return b;
        }
        return std::nullopt;
    }

    bool cancelBooking(int bookingId) override {
        return bookingId == 1001;
    }

    std::vector<Booking> getBookingsByUser(int userId) override {
        return {};
    }

    std::vector<Booking> getBookingsByShow(int showId) override {
        return {};
    }
};

class MockPaymentService : public IPaymentService {
public:
    PaymentResultDTO processPayment(int bookingId, double amount, bool failMock) override {
        if (bookingId <= 0 || amount <= 0.0) {
            return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Invalid arguments", 400};
        }
        if (bookingId == 999) {
            return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Booking not found", 404};
        }
        if (failMock) {
            return {false, 5002, bookingId, PaymentStatus::FAILED, std::nullopt, "Mock payment failure requested", 400};
        }
        return {true, 5001, bookingId, PaymentStatus::SUCCESS, "TXN-1001-12345", "", 200};
    }

    std::optional<Payment> getPayment(int paymentId) override {
        return std::nullopt;
    }

    std::optional<Payment> getPaymentByBooking(int bookingId) override {
        return std::nullopt;
    }
};

// --- Test Suites ---

TEST(ShowControllerTest, GetAllShowsReturnsList) {
    auto mockShowService = std::make_shared<MockShowService>();
    Show s;
    s.id = 1;
    s.movie_id = 10;
    s.screen_id = 20;
    s.start_time = "2026-08-15 10:00:00";
    s.end_time = "2026-08-15 12:00:00";
    mockShowService->mockShows.push_back(s);

    ShowController controller(mockShowService);
    crow::request req;
    auto res = controller.handleGetAllShows(req);

    EXPECT_EQ(res.code, 200);
    auto body = crow::json::load(res.body);
    ASSERT_TRUE(body);
    EXPECT_EQ(body.size(), 1u);
    EXPECT_EQ(body[0]["id"].i(), 1);
    EXPECT_EQ(body[0]["movie_id"].i(), 10);
}

TEST(ShowControllerTest, GetShowSeatsSuccessAndNotFound) {
    auto mockShowService = std::make_shared<MockShowService>();
    Show s;
    s.id = 1;
    s.movie_id = 10;
    s.screen_id = 20;
    mockShowService->mockShows.push_back(s);

    ShowSeat ss;
    ss.id = 1;
    ss.show_id = 1;
    ss.seat_id = 1;
    ss.status = SeatStatus::AVAILABLE;
    mockShowService->mockSeats.push_back(ss);

    ShowController controller(mockShowService);
    crow::request req;

    // Valid show
    auto res = controller.handleGetShowSeats(req, 1);
    EXPECT_EQ(res.code, 200);
    auto body = crow::json::load(res.body);
    ASSERT_TRUE(body);
    EXPECT_EQ(body.size(), 1u);
    EXPECT_EQ(body[0]["seat_id"].i(), 1);
    EXPECT_EQ(body[0]["status"].s(), "AVAILABLE");

    // Non-existent show -> 404
    auto resNotFound = controller.handleGetShowSeats(req, 999);
    EXPECT_EQ(resNotFound.code, 404);

    // Negative show -> 404
    auto resInvalid = controller.handleGetShowSeats(req, -1);
    EXPECT_EQ(resInvalid.code, 404);
}

TEST(BookingControllerTest, CreateBookingSuccess) {
    auto mockBookingService = std::make_shared<MockBookingService>();
    BookingController controller(mockBookingService);

    crow::request req;
    req.body = R"({"show_id": 1, "user_id": 1, "seat_ids": [1, 2]})";

    auto res = controller.handleCreateBooking(req);
    EXPECT_EQ(res.code, 201);
    auto body = crow::json::load(res.body);
    ASSERT_TRUE(body);
    EXPECT_EQ(body["booking_id"].i(), 1001);
    EXPECT_EQ(body["status"].s(), "SEATS_LOCKED");
    EXPECT_DOUBLE_EQ(body["total_price"].d(), 50.0);
}

TEST(BookingControllerTest, CreateBookingValidationErrors) {
    auto mockBookingService = std::make_shared<MockBookingService>();
    BookingController controller(mockBookingService);

    // Invalid JSON
    {
        crow::request req;
        req.body = "invalid-json";
        auto res = controller.handleCreateBooking(req);
        EXPECT_EQ(res.code, 400);
    }

    // Missing fields
    {
        crow::request req;
        req.body = R"({"show_id": 1})";
        auto res = controller.handleCreateBooking(req);
        EXPECT_EQ(res.code, 400);
    }

    // Empty seat_ids
    {
        crow::request req;
        req.body = R"({"show_id": 1, "user_id": 1, "seat_ids": []})";
        auto res = controller.handleCreateBooking(req);
        EXPECT_EQ(res.code, 400);
    }

    // Show not found -> 404
    {
        crow::request req;
        req.body = R"({"show_id": 999, "user_id": 1, "seat_ids": [1]})";
        auto res = controller.handleCreateBooking(req);
        EXPECT_EQ(res.code, 404);
    }

    // Seat already locked -> 409
    {
        crow::request req;
        req.body = R"({"show_id": 1, "user_id": 1, "seat_ids": [99]})";
        auto res = controller.handleCreateBooking(req);
        EXPECT_EQ(res.code, 409);
    }
}

TEST(BookingControllerTest, GetBookingById) {
    auto mockBookingService = std::make_shared<MockBookingService>();
    BookingController controller(mockBookingService);
    crow::request req;

    // Existing booking
    auto res = controller.handleGetBooking(req, 1001);
    EXPECT_EQ(res.code, 200);
    auto body = crow::json::load(res.body);
    ASSERT_TRUE(body);
    EXPECT_EQ(body["booking_id"].i(), 1001);
    EXPECT_EQ(body["status"].s(), "SEATS_LOCKED");

    // Non-existent booking -> 404
    auto res404 = controller.handleGetBooking(req, 9999);
    EXPECT_EQ(res404.code, 404);

    // Negative ID -> 404
    auto resNeg = controller.handleGetBooking(req, -5);
    EXPECT_EQ(resNeg.code, 404);
}

TEST(PaymentControllerTest, ProcessPaymentSuccessAndFailure) {
    auto mockPaymentService = std::make_shared<MockPaymentService>();
    PaymentController controller(mockPaymentService);

    // Success (fail_mock: false)
    {
        crow::request req;
        req.body = R"({"booking_id": 1001, "amount": 50.0, "fail_mock": false})";
        auto res = controller.handleProcessPayment(req);
        EXPECT_EQ(res.code, 200);
        auto body = crow::json::load(res.body);
        ASSERT_TRUE(body);
        EXPECT_EQ(body["payment_id"].i(), 5001);
        EXPECT_EQ(body["status"].s(), "SUCCESS");
    }

    // Failure simulation (fail_mock: true)
    {
        crow::request req;
        req.body = R"({"booking_id": 1001, "amount": 50.0, "fail_mock": true})";
        auto res = controller.handleProcessPayment(req);
        EXPECT_EQ(res.code, 400);
        auto body = crow::json::load(res.body);
        ASSERT_TRUE(body);
        EXPECT_EQ(body["status"].s(), "FAILED");
    }

    // Booking not found -> 404
    {
        crow::request req;
        req.body = R"({"booking_id": 999, "amount": 50.0, "fail_mock": false})";
        auto res = controller.handleProcessPayment(req);
        EXPECT_EQ(res.code, 404);
    }

    // Invalid body
    {
        crow::request req;
        req.body = R"({"amount": 50.0})";
        auto res = controller.handleProcessPayment(req);
        EXPECT_EQ(res.code, 400);
    }
}

TEST(ResetControllerTest, NullDatabaseReturns500) {
    ResetController controller(nullptr);
    crow::request req;
    auto res = controller.handleReset(req);
    EXPECT_EQ(res.code, 500);
    auto body = crow::json::load(res.body);
    ASSERT_TRUE(body);
    EXPECT_EQ(body["status"].s(), "ERROR");
}
