#pragma once
#include "IPaymentGateway.h"
#include <atomic>
#include <cstdint>

class MockPaymentGateway : public IPaymentGateway {
public:
    MockPaymentGateway() = default;
    PaymentResult processPayment(int bookingId, double amount, bool failMock) override;

private:
    static std::atomic<uint64_t> txnCounter_;
};
