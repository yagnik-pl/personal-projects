#pragma once
#include "BookingState.h"

class ConfirmedState : public BookingState {
public:
    std::string getStatusName() const override;
    bool canTransitionTo(const std::string& targetStatus) const override;
};
