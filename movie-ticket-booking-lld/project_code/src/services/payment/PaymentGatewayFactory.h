#pragma once
#include <memory>
#include <string>
#include "IPaymentGateway.h"

class PaymentGatewayFactory {
public:
    static std::shared_ptr<IPaymentGateway> create(const std::string& gatewayType = "MOCK");
    static std::shared_ptr<IPaymentGateway> createGateway(const std::string& gatewayType = "MOCK") {
        return create(gatewayType);
    }
};
