#pragma once
#include <crow.h>
#include <memory>
#include "../services/IPaymentService.h"

class PaymentController {
public:
    explicit PaymentController(std::shared_ptr<IPaymentService> paymentService);

    void registerRoutes(crow::SimpleApp& app);

    crow::response handleProcessPayment(const crow::request& req);

private:
    std::shared_ptr<IPaymentService> paymentService_;
};
