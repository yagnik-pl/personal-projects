#pragma once
#include "BookingState.h"

class SeatsLockedState : public BookingState {
public:
    std::string getStatusName() const override;
    bool canTransitionTo(const std::string& targetStatus) const override;

    void lockSeats(BookingContext& context) override;
    void initiatePayment(BookingContext& context) override;
    void confirmPayment(BookingContext& context) override;
    void cancel(BookingContext& context) override;
    void failPayment(BookingContext& context) override;
};
