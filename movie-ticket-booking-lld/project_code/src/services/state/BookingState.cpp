#include "BookingState.h"
#include "CreatedState.h"
#include "SeatsLockedState.h"
#include "PaymentPendingState.h"
#include "ConfirmedState.h"
#include "CancelledState.h"

void BookingState::lockSeats(BookingContext& context) {
    throw InvalidStateTransitionException("Cannot lock seats from state " + getStatusName());
}

void BookingState::initiatePayment(BookingContext& context) {
    throw InvalidStateTransitionException("Cannot initiate payment from state " + getStatusName());
}

void BookingState::confirmPayment(BookingContext& context) {
    throw InvalidStateTransitionException("Cannot confirm payment from state " + getStatusName());
}

void BookingState::cancel(BookingContext& context) {
    throw InvalidStateTransitionException("Cannot cancel booking from state " + getStatusName());
}

void BookingState::failPayment(BookingContext& context) {
    throw InvalidStateTransitionException("Cannot fail payment from state " + getStatusName());
}

std::shared_ptr<BookingState> BookingState::fromString(const std::string& status) {
    if (status == BookingStatus::CREATED) {
        return std::make_shared<CreatedState>();
    } else if (status == BookingStatus::SEATS_LOCKED) {
        return std::make_shared<SeatsLockedState>();
    } else if (status == BookingStatus::PAYMENT_PENDING) {
        return std::make_shared<PaymentPendingState>();
    } else if (status == BookingStatus::CONFIRMED) {
        return std::make_shared<ConfirmedState>();
    } else if (status == BookingStatus::CANCELLED) {
        return std::make_shared<CancelledState>();
    }
    throw InvalidArgumentException("Unknown booking status: " + status);
}

// BookingContext implementation
BookingContext::BookingContext(int bookingId, std::shared_ptr<BookingState> initialState)
    : bookingId_(bookingId), state_(std::move(initialState)) {
    if (!state_) {
        throw InvalidArgumentException("Initial state cannot be null");
    }
}

BookingContext::BookingContext(int bookingId, const std::string& initialStatus)
    : bookingId_(bookingId), state_(BookingState::fromString(initialStatus)) {}

std::string BookingContext::getStatusName() const {
    return state_->getStatusName();
}

void BookingContext::setState(std::shared_ptr<BookingState> newState) {
    if (!newState) {
        throw InvalidArgumentException("New state cannot be null");
    }
    state_ = std::move(newState);
}

void BookingContext::lockSeats() {
    state_->lockSeats(*this);
}

void BookingContext::initiatePayment() {
    state_->initiatePayment(*this);
}

void BookingContext::confirmPayment() {
    state_->confirmPayment(*this);
}

void BookingContext::cancel() {
    state_->cancel(*this);
}

void BookingContext::failPayment() {
    state_->failPayment(*this);
}
