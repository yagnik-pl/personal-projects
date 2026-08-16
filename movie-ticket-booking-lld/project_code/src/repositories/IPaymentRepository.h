#pragma once
#include <optional>
#include <vector>
#include <string>
#include "../models/Payment.h"

class IPaymentRepository {
public:
    virtual ~IPaymentRepository() = default;
    virtual Payment createPayment(const Payment& payment) = 0;
    virtual std::optional<Payment> getPayment(int id) = 0;
    virtual std::optional<Payment> getPaymentByBookingId(int bookingId) = 0;
    virtual bool updatePaymentStatus(int id, const std::string& status, const std::optional<std::string>& transactionId = std::nullopt) = 0;
};
