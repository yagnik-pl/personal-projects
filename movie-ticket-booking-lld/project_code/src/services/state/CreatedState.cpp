#include "CreatedState.h"
#include "SeatsLockedState.h"
#include "CancelledState.h"

std::string CreatedState::getStatusName() const {
    return BookingStatus::CREATED;
}

bool CreatedState::canTransitionTo(const std::string& targetStatus) const {
    return targetStatus == BookingStatus::SEATS_LOCKED ||
           targetStatus == BookingStatus::CANCELLED;
}

void CreatedState::lockSeats(BookingContext& context) {
    context.setState(std::make_shared<SeatsLockedState>());
}

void CreatedState::cancel(BookingContext& context) {
    context.setState(std::make_shared<CancelledState>());
}
