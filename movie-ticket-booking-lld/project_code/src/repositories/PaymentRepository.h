#pragma once
#include "IPaymentRepository.h"
#include "../core/Database.h"
#include <memory>

class PaymentRepository : public IPaymentRepository {
public:
    explicit PaymentRepository(std::shared_ptr<Database> db);
    Payment createPayment(const Payment& payment) override;
    std::optional<Payment> getPayment(int id) override;
    std::optional<Payment> getPaymentByBookingId(int bookingId) override;
    bool updatePaymentStatus(int id, const std::string& status, const std::optional<std::string>& transactionId = std::nullopt) override;

private:
    std::shared_ptr<Database> db_;
};
