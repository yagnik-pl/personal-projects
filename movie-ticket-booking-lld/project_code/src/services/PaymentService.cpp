#include "PaymentService.h"
#include "state/BookingState.h"
#include "../models/Enums.h"
#include <cmath>

PaymentService::PaymentService(
    std::shared_ptr<IPaymentRepository> paymentRepo,
    std::shared_ptr<IBookingRepository> bookingRepo,
    std::shared_ptr<IShowSeatRepository> showSeatRepo,
    std::shared_ptr<IPaymentGateway> paymentGateway
) : paymentRepo_(paymentRepo),
    bookingRepo_(bookingRepo),
    showSeatRepo_(showSeatRepo),
    paymentGateway_(paymentGateway) {}

PaymentResultDTO PaymentService::processPayment(int bookingId, double amount, bool failMock) {
    if (bookingId <= 0) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Invalid booking_id: booking_id must be a positive integer", 400};
    }

    if (amount <= 0.0) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Invalid payment amount: amount must be positive", 400};
    }

    auto bookingOpt = bookingRepo_->getBooking(bookingId);
    if (!bookingOpt.has_value()) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Booking not found", 404};
    }

    const auto& booking = *bookingOpt;

    if (booking.status == BookingStatus::CONFIRMED) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Booking is already CONFIRMED", 409};
    }

    if (booking.status == BookingStatus::CANCELLED) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Booking is CANCELLED and cannot be paid", 409};
    }

    if (booking.status != BookingStatus::SEATS_LOCKED && booking.status != BookingStatus::PAYMENT_PENDING) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Booking is not in a payable state: " + booking.status, 409};
    }

    if (std::abs(amount - booking.amount) > 0.001) {
        return {false, std::nullopt, bookingId, PaymentStatus::FAILED, std::nullopt, "Payment amount does not match booking total amount", 400};
    }

    BookingContext context(bookingId, booking.status);

    PaymentResult gatewayResult = paymentGateway_->processPayment(bookingId, amount, failMock);

    if (!gatewayResult.success) {
        Payment payment;
        payment.booking_id = bookingId;
        payment.status = PaymentStatus::FAILED;

        Payment createdPayment = paymentRepo_->createPayment(payment);
        try {
            context.failPayment();
        } catch (...) {}

        std::string err = gatewayResult.errorMessage.empty() ? "Payment processing failed" : gatewayResult.errorMessage;
        return {false, createdPayment.id, bookingId, PaymentStatus::FAILED, std::nullopt, err, 400};
    }

    // Payment succeeded
    Payment payment;
    payment.booking_id = bookingId;
    payment.status = PaymentStatus::SUCCESS;
    payment.transaction_id = gatewayResult.transactionId;

    Payment createdPayment = paymentRepo_->createPayment(payment);

    context.confirmPayment();
    bookingRepo_->updateBookingStatus(bookingId, BookingStatus::CONFIRMED);
    showSeatRepo_->confirmSeats(bookingId);

    return {true, createdPayment.id, bookingId, PaymentStatus::SUCCESS, gatewayResult.transactionId, "", 200};
}

std::optional<Payment> PaymentService::getPayment(int paymentId) {
    if (paymentId <= 0) {
        return std::nullopt;
    }
    return paymentRepo_->getPayment(paymentId);
}

std::optional<Payment> PaymentService::getPaymentByBooking(int bookingId) {
    if (bookingId <= 0) {
        return std::nullopt;
    }
    return paymentRepo_->getPaymentByBookingId(bookingId);
}
