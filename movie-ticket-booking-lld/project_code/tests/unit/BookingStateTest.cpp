#include <gtest/gtest.h>
#include "services/state/BookingState.h"
#include "services/state/CreatedState.h"
#include "services/state/SeatsLockedState.h"
#include "services/state/PaymentPendingState.h"
#include "services/state/ConfirmedState.h"
#include "services/state/CancelledState.h"
#include "models/Enums.h"
#include "core/Exceptions.h"

TEST(BookingStateTest, CreatedStateTransitions) {
    BookingContext ctx(1, std::make_shared<CreatedState>());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CREATED);

    auto state = ctx.getState();
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::SEATS_LOCKED));
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::CANCELLED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CONFIRMED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::PAYMENT_PENDING));

    // Valid transitions
    EXPECT_NO_THROW(ctx.lockSeats());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::SEATS_LOCKED);
}

TEST(BookingStateTest, CreatedStateCancel) {
    BookingContext ctx(2, std::make_shared<CreatedState>());
    EXPECT_NO_THROW(ctx.cancel());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CANCELLED);
}

TEST(BookingStateTest, CreatedStateInvalidTransitions) {
    BookingContext ctx(3, std::make_shared<CreatedState>());
    EXPECT_THROW(ctx.initiatePayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.confirmPayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.failPayment(), InvalidStateTransitionException);
}

TEST(BookingStateTest, SeatsLockedStateTransitions) {
    BookingContext ctx(4, std::make_shared<SeatsLockedState>());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::SEATS_LOCKED);

    auto state = ctx.getState();
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::PAYMENT_PENDING));
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::CONFIRMED));
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::CANCELLED));

    // Initiate payment
    EXPECT_NO_THROW(ctx.initiatePayment());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::PAYMENT_PENDING);
}

TEST(BookingStateTest, SeatsLockedDirectConfirm) {
    BookingContext ctx(5, std::make_shared<SeatsLockedState>());
    EXPECT_NO_THROW(ctx.confirmPayment());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CONFIRMED);
}

TEST(BookingStateTest, SeatsLockedCancel) {
    BookingContext ctx(6, std::make_shared<SeatsLockedState>());
    EXPECT_NO_THROW(ctx.cancel());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CANCELLED);
}

TEST(BookingStateTest, PaymentPendingStateTransitions) {
    BookingContext ctx(7, std::make_shared<PaymentPendingState>());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::PAYMENT_PENDING);

    auto state = ctx.getState();
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::CONFIRMED));
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::SEATS_LOCKED));
    EXPECT_TRUE(state->canTransitionTo(BookingStatus::CANCELLED));

    // Confirm payment
    EXPECT_NO_THROW(ctx.confirmPayment());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CONFIRMED);
}

TEST(BookingStateTest, PaymentPendingFailRetry) {
    BookingContext ctx(8, std::make_shared<PaymentPendingState>());
    EXPECT_NO_THROW(ctx.failPayment());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::SEATS_LOCKED);
}

TEST(BookingStateTest, ConfirmedStateTerminal) {
    BookingContext ctx(9, std::make_shared<ConfirmedState>());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CONFIRMED);

    auto state = ctx.getState();
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CREATED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::SEATS_LOCKED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::PAYMENT_PENDING));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CONFIRMED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CANCELLED));

    EXPECT_THROW(ctx.lockSeats(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.initiatePayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.confirmPayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.cancel(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.failPayment(), InvalidStateTransitionException);
}

TEST(BookingStateTest, CancelledStateTerminal) {
    BookingContext ctx(10, std::make_shared<CancelledState>());
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::CANCELLED);

    auto state = ctx.getState();
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CREATED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::SEATS_LOCKED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::PAYMENT_PENDING));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CONFIRMED));
    EXPECT_FALSE(state->canTransitionTo(BookingStatus::CANCELLED));

    EXPECT_THROW(ctx.lockSeats(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.initiatePayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.confirmPayment(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.cancel(), InvalidStateTransitionException);
    EXPECT_THROW(ctx.failPayment(), InvalidStateTransitionException);
}

TEST(BookingStateTest, FromStringFactory) {
    EXPECT_EQ(BookingState::fromString("CREATED")->getStatusName(), BookingStatus::CREATED);
    EXPECT_EQ(BookingState::fromString("SEATS_LOCKED")->getStatusName(), BookingStatus::SEATS_LOCKED);
    EXPECT_EQ(BookingState::fromString("PAYMENT_PENDING")->getStatusName(), BookingStatus::PAYMENT_PENDING);
    EXPECT_EQ(BookingState::fromString("CONFIRMED")->getStatusName(), BookingStatus::CONFIRMED);
    EXPECT_EQ(BookingState::fromString("CANCELLED")->getStatusName(), BookingStatus::CANCELLED);

    EXPECT_THROW(BookingState::fromString("INVALID_STATUS"), InvalidArgumentException);
    EXPECT_THROW(BookingState::fromString(""), InvalidArgumentException);
}

TEST(BookingStateTest, ContextConstructorWithString) {
    BookingContext ctx(100, BookingStatus::SEATS_LOCKED);
    EXPECT_EQ(ctx.getBookingId(), 100);
    EXPECT_EQ(ctx.getStatusName(), BookingStatus::SEATS_LOCKED);
}
