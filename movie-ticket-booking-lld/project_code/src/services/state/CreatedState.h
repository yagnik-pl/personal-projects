#pragma once
#include "BookingState.h"

class CreatedState : public BookingState {
public:
    std::string getStatusName() const override;
    bool canTransitionTo(const std::string& targetStatus) const override;

    void lockSeats(BookingContext& context) override;
    void cancel(BookingContext& context) override;
};
