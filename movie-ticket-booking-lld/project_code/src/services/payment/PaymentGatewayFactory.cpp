#include "PaymentGatewayFactory.h"
#include "MockPaymentGateway.h"
#include "../../core/Exceptions.h"
#include <algorithm>

std::shared_ptr<IPaymentGateway> PaymentGatewayFactory::create(const std::string& gatewayType) {
    std::string typeUpper = gatewayType;
    std::transform(typeUpper.begin(), typeUpper.end(), typeUpper.begin(), ::toupper);

    if (typeUpper.empty() || typeUpper == "MOCK") {
        return std::make_shared<MockPaymentGateway>();
    }

    throw InvalidArgumentException("Unknown payment gateway type: " + gatewayType);
}
