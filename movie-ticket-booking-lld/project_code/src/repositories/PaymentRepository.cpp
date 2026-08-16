#include "PaymentRepository.h"
#include <pqxx/pqxx>

PaymentRepository::PaymentRepository(std::shared_ptr<Database> db) : db_(db) {}

Payment PaymentRepository::createPayment(const Payment& payment) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    std::string status = payment.status.empty() ? "PENDING" : payment.status;

    pqxx::result r = w.exec_params(
        "INSERT INTO payments (booking_id, status, transaction_id) "
        "VALUES ($1, $2::payment_status, $3) "
        "RETURNING id, created_at::text",
        payment.booking_id, status, payment.transaction_id
    );
    w.commit();

    Payment created = payment;
    created.id = r[0]["id"].as<int>();
    created.status = status;
    if (!r[0]["created_at"].is_null()) {
        created.created_at = r[0]["created_at"].as<std::string>();
    }
    return created;
}

std::optional<Payment> PaymentRepository::getPayment(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, booking_id, status::text, transaction_id, created_at::text FROM payments WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Payment payment;
    payment.id = r[0]["id"].as<int>();
    payment.booking_id = r[0]["booking_id"].as<int>();
    payment.status = r[0]["status"].as<std::string>();
    if (!r[0]["transaction_id"].is_null()) {
        payment.transaction_id = r[0]["transaction_id"].as<std::string>();
    }
    if (!r[0]["created_at"].is_null()) {
        payment.created_at = r[0]["created_at"].as<std::string>();
    }

    return payment;
}

std::optional<Payment> PaymentRepository::getPaymentByBookingId(int bookingId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, booking_id, status::text, transaction_id, created_at::text "
        "FROM payments WHERE booking_id = $1 ORDER BY id DESC LIMIT 1",
        bookingId
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Payment payment;
    payment.id = r[0]["id"].as<int>();
    payment.booking_id = r[0]["booking_id"].as<int>();
    payment.status = r[0]["status"].as<std::string>();
    if (!r[0]["transaction_id"].is_null()) {
        payment.transaction_id = r[0]["transaction_id"].as<std::string>();
    }
    if (!r[0]["created_at"].is_null()) {
        payment.created_at = r[0]["created_at"].as<std::string>();
    }

    return payment;
}

bool PaymentRepository::updatePaymentStatus(int id, const std::string& status, const std::optional<std::string>& transactionId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r;
    if (transactionId.has_value()) {
        r = w.exec_params(
            "UPDATE payments SET status = $1::payment_status, transaction_id = $2 WHERE id = $3",
            status, *transactionId, id
        );
    } else {
        r = w.exec_params(
            "UPDATE payments SET status = $1::payment_status WHERE id = $2",
            status, id
        );
    }
    w.commit();
    return r.affected_rows() > 0;
}
