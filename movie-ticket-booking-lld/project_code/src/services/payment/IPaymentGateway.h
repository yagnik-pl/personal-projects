#pragma once
#include <string>

struct PaymentResult {
    bool success{false};
    std::string transactionId;
    std::string errorMessage;
};

class IPaymentGateway {
public:
    virtual ~IPaymentGateway() = default;
    virtual PaymentResult processPayment(int bookingId, double amount, bool failMock) = 0;
};
