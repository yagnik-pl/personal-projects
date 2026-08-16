#pragma once
#include "BookingState.h"

class PaymentPendingState : public BookingState {
public:
    std::string getStatusName() const override;
    bool canTransitionTo(const std::string& targetStatus) const override;

    void confirmPayment(BookingContext& context) override;
    void failPayment(BookingContext& context) override;
    void cancel(BookingContext& context) override;
};
