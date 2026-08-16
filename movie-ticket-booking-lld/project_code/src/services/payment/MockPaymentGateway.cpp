#include "MockPaymentGateway.h"
#include <chrono>
#include <sstream>
#include <iomanip>

std::atomic<uint64_t> MockPaymentGateway::txnCounter_{1000};

PaymentResult MockPaymentGateway::processPayment(int bookingId, double amount, bool failMock) {
    if (failMock) {
        return {false, "", "Mock payment failure requested"};
    }

    if (amount <= 0.0) {
        return {false, "", "Invalid payment amount: amount must be positive"};
    }

    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    uint64_t seq = txnCounter_.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "TXN-" << bookingId << "-" << nowMs << "-" << seq;

    return {true, oss.str(), ""};
}
