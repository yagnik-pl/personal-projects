#include "SeatsLockedState.h"
#include "PaymentPendingState.h"
#include "ConfirmedState.h"
#include "CancelledState.h"

std::string SeatsLockedState::getStatusName() const {
    return BookingStatus::SEATS_LOCKED;
}

bool SeatsLockedState::canTransitionTo(const std::string& targetStatus) const {
    return targetStatus == BookingStatus::PAYMENT_PENDING ||
           targetStatus == BookingStatus::CONFIRMED ||
           targetStatus == BookingStatus::CANCELLED ||
           targetStatus == BookingStatus::SEATS_LOCKED;
}

void SeatsLockedState::lockSeats(BookingContext& context) {
    // Already in SeatsLockedState, no-op
}

void SeatsLockedState::initiatePayment(BookingContext& context) {
    context.setState(std::make_shared<PaymentPendingState>());
}

void SeatsLockedState::confirmPayment(BookingContext& context) {
    context.setState(std::make_shared<ConfirmedState>());
}

void SeatsLockedState::cancel(BookingContext& context) {
    context.setState(std::make_shared<CancelledState>());
}

void SeatsLockedState::failPayment(BookingContext& context) {
    // Stays in SeatsLockedState allowing retry until expiration
}
