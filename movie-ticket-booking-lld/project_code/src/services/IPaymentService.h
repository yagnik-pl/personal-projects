#pragma once
#include <optional>
#include <string>
#include "../models/Payment.h"

struct PaymentResultDTO {
    bool success{false};
    std::optional<int> paymentId;
    std::optional<int> bookingId;
    std::string status;
    std::optional<std::string> transactionId;
    std::string errorMessage;
    int httpStatusCode{200};
};

class IPaymentService {
public:
    virtual ~IPaymentService() = default;

    virtual PaymentResultDTO processPayment(int bookingId, double amount, bool failMock) = 0;
    virtual std::optional<Payment> getPayment(int paymentId) = 0;
    virtual std::optional<Payment> getPaymentByBooking(int bookingId) = 0;
};
