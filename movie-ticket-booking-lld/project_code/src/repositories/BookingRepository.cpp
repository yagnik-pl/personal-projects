#include "BookingRepository.h"
#include <pqxx/pqxx>

BookingRepository::BookingRepository(std::shared_ptr<Database> db) : db_(db) {}

Booking BookingRepository::createBooking(const Booking& booking) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    std::string status = booking.status.empty() ? "CREATED" : booking.status;
    pqxx::result r = w.exec_params(
        "INSERT INTO bookings (user_id, show_id, status, amount) "
        "VALUES ($1, $2, $3::booking_status, $4) "
        "RETURNING id, created_at::text",
        booking.user_id, booking.show_id, status, booking.amount
    );
    w.commit();

    Booking created = booking;
    created.id = r[0]["id"].as<int>();
    created.status = status;
    created.created_at = r[0]["created_at"].as<std::string>();
    return created;
}

std::optional<Booking> BookingRepository::getBooking(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, user_id, show_id, status::text, amount, created_at::text FROM bookings WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Booking booking;
    booking.id = r[0]["id"].as<int>();
    booking.user_id = r[0]["user_id"].as<int>();
    booking.show_id = r[0]["show_id"].as<int>();
    booking.status = r[0]["status"].as<std::string>();
    booking.amount = r[0]["amount"].as<double>();
    if (!r[0]["created_at"].is_null()) {
        booking.created_at = r[0]["created_at"].as<std::string>();
    }

    return booking;
}

bool BookingRepository::updateBookingStatus(int id, const std::string& status) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "UPDATE bookings SET status = $1::booking_status WHERE id = $2",
        status, id
    );
    w.commit();
    return r.affected_rows() > 0;
}

bool BookingRepository::updateBookingAmount(int id, double amount) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "UPDATE bookings SET amount = $1 WHERE id = $2",
        amount, id
    );
    w.commit();
    return r.affected_rows() > 0;
}

std::vector<Booking> BookingRepository::getBookingsByUser(int userId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, user_id, show_id, status::text, amount, created_at::text FROM bookings WHERE user_id = $1 ORDER BY id DESC",
        userId
    );

    std::vector<Booking> bookings;
    bookings.reserve(r.size());

    for (const auto& row : r) {
        Booking booking;
        booking.id = row["id"].as<int>();
        booking.user_id = row["user_id"].as<int>();
        booking.show_id = row["show_id"].as<int>();
        booking.status = row["status"].as<std::string>();
        booking.amount = row["amount"].as<double>();
        if (!row["created_at"].is_null()) {
            booking.created_at = row["created_at"].as<std::string>();
        }
        bookings.push_back(booking);
    }

    return bookings;
}

std::vector<Booking> BookingRepository::getBookingsByShow(int showId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, user_id, show_id, status::text, amount, created_at::text FROM bookings WHERE show_id = $1 ORDER BY id ASC",
        showId
    );

    std::vector<Booking> bookings;
    bookings.reserve(r.size());

    for (const auto& row : r) {
        Booking booking;
        booking.id = row["id"].as<int>();
        booking.user_id = row["user_id"].as<int>();
        booking.show_id = row["show_id"].as<int>();
        booking.status = row["status"].as<std::string>();
        booking.amount = row["amount"].as<double>();
        if (!row["created_at"].is_null()) {
            booking.created_at = row["created_at"].as<std::string>();
        }
        bookings.push_back(booking);
    }

    return bookings;
}
