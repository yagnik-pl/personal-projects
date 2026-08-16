#include "CancelledState.h"

std::string CancelledState::getStatusName() const {
    return BookingStatus::CANCELLED;
}

bool CancelledState::canTransitionTo(const std::string& targetStatus) const {
    return false; // Terminal state
}
