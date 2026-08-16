#include "PaymentPendingState.h"
#include "SeatsLockedState.h"
#include "ConfirmedState.h"
#include "CancelledState.h"

std::string PaymentPendingState::getStatusName() const {
    return BookingStatus::PAYMENT_PENDING;
}

bool PaymentPendingState::canTransitionTo(const std::string& targetStatus) const {
    return targetStatus == BookingStatus::CONFIRMED ||
           targetStatus == BookingStatus::SEATS_LOCKED ||
           targetStatus == BookingStatus::CANCELLED ||
           targetStatus == BookingStatus::PAYMENT_PENDING;
}

void PaymentPendingState::confirmPayment(BookingContext& context) {
    context.setState(std::make_shared<ConfirmedState>());
}

void PaymentPendingState::failPayment(BookingContext& context) {
    context.setState(std::make_shared<SeatsLockedState>());
}

void PaymentPendingState::cancel(BookingContext& context) {
    context.setState(std::make_shared<CancelledState>());
}
