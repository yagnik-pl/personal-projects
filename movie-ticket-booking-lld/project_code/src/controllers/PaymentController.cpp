#include "PaymentController.h"

PaymentController::PaymentController(std::shared_ptr<IPaymentService> paymentService)
    : paymentService_(paymentService) {}

void PaymentController::registerRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/payments")
    .methods(crow::HTTPMethod::POST)
    ([this](const crow::request& req) {
        return this->handleProcessPayment(req);
    });
}

crow::response PaymentController::handleProcessPayment(const crow::request& req) {
    auto body = crow::json::load(req.body);
    if (!body) {
        crow::json::wvalue res;
        res["status"] = "FAILED";
        res["error"] = "Invalid JSON body";
        return crow::response(400, res);
    }

    if (!body.has("booking_id") || !body.has("amount")) {
        crow::json::wvalue res;
        res["status"] = "FAILED";
        res["error"] = "Missing required fields: booking_id and amount";
        return crow::response(400, res);
    }

    if (body["booking_id"].t() != crow::json::type::Number ||
        body["amount"].t() != crow::json::type::Number) {
        crow::json::wvalue res;
        res["status"] = "FAILED";
        res["error"] = "booking_id and amount must be numeric values";
        return crow::response(400, res);
    }

    int bookingId = static_cast<int>(body["booking_id"].i());
    double amount = body["amount"].d();

    bool failMock = false;
    if (body.has("fail_mock")) {
        if (body["fail_mock"].t() == crow::json::type::True) {
            failMock = true;
        } else if (body["fail_mock"].t() == crow::json::type::False) {
            failMock = false;
        } else if (body["fail_mock"].t() == crow::json::type::Number) {
            failMock = (body["fail_mock"].i() != 0);
        } else if (body["fail_mock"].t() == crow::json::type::String) {
            std::string s = body["fail_mock"].s();
            failMock = (s == "true" || s == "True" || s == "1");
        }
    }

    PaymentResultDTO result = paymentService_->processPayment(bookingId, amount, failMock);

    if (result.success) {
        crow::json::wvalue res;
        if (result.paymentId.has_value()) {
            res["payment_id"] = *result.paymentId;
        }
        res["status"] = "SUCCESS";
        if (result.transactionId.has_value()) {
            res["transaction_id"] = *result.transactionId;
        }
        return crow::response(200, res);
    } else {
        crow::json::wvalue res;
        res["status"] = "FAILED";
        res["error"] = result.errorMessage;
        if (result.paymentId.has_value()) {
            res["payment_id"] = *result.paymentId;
        }
        return crow::response(result.httpStatusCode, res);
    }
}
