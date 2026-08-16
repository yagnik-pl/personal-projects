#include "ConfirmedState.h"

std::string ConfirmedState::getStatusName() const {
    return BookingStatus::CONFIRMED;
}

bool ConfirmedState::canTransitionTo(const std::string& targetStatus) const {
    return false; // Terminal state
}
