#pragma once
#include "IPaymentService.h"
#include "payment/IPaymentGateway.h"
#include "../repositories/IPaymentRepository.h"
#include "../repositories/IBookingRepository.h"
#include "../repositories/IShowSeatRepository.h"
#include <memory>

class PaymentService : public IPaymentService {
public:
    PaymentService(
        std::shared_ptr<IPaymentRepository> paymentRepo,
        std::shared_ptr<IBookingRepository> bookingRepo,
        std::shared_ptr<IShowSeatRepository> showSeatRepo,
        std::shared_ptr<IPaymentGateway> paymentGateway
    );

    PaymentResultDTO processPayment(int bookingId, double amount, bool failMock) override;
    std::optional<Payment> getPayment(int paymentId) override;
    std::optional<Payment> getPaymentByBooking(int bookingId) override;

private:
    std::shared_ptr<IPaymentRepository> paymentRepo_;
    std::shared_ptr<IBookingRepository> bookingRepo_;
    std::shared_ptr<IShowSeatRepository> showSeatRepo_;
    std::shared_ptr<IPaymentGateway> paymentGateway_;
};
