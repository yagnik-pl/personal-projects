#include <gtest/gtest.h>
#include "services/payment/MockPaymentGateway.h"
#include "services/payment/PaymentGatewayFactory.h"
#include "core/Exceptions.h"

TEST(PaymentGatewayTest, MockPaymentGatewaySuccess) {
    MockPaymentGateway gateway;
    PaymentResult result = gateway.processPayment(1001, 50.0, false);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.transactionId.empty());
    EXPECT_TRUE(result.transactionId.rfind("TXN-1001-", 0) == 0);
    EXPECT_TRUE(result.errorMessage.empty());
}

TEST(PaymentGatewayTest, MockPaymentGatewayFailureFlag) {
    MockPaymentGateway gateway;
    PaymentResult result = gateway.processPayment(1002, 25.0, true);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.transactionId.empty());
    EXPECT_EQ(result.errorMessage, "Mock payment failure requested");
}

TEST(PaymentGatewayTest, MockPaymentGatewayInvalidAmount) {
    MockPaymentGateway gateway;
    PaymentResult resultZero = gateway.processPayment(1003, 0.0, false);
    EXPECT_FALSE(resultZero.success);

    PaymentResult resultNegative = gateway.processPayment(1004, -10.0, false);
    EXPECT_FALSE(resultNegative.success);
}

TEST(PaymentGatewayTest, MockPaymentGatewayUniqueTransactions) {
    MockPaymentGateway gateway;
    PaymentResult res1 = gateway.processPayment(1005, 50.0, false);
    PaymentResult res2 = gateway.processPayment(1005, 50.0, false);

    EXPECT_TRUE(res1.success);
    EXPECT_TRUE(res2.success);
    EXPECT_NE(res1.transactionId, res2.transactionId);
}

TEST(PaymentGatewayTest, FactoryCreation) {
    auto gateway1 = PaymentGatewayFactory::create("MOCK");
    EXPECT_NE(gateway1, nullptr);

    auto gateway2 = PaymentGatewayFactory::create("mock");
    EXPECT_NE(gateway2, nullptr);

    auto gateway3 = PaymentGatewayFactory::create("");
    EXPECT_NE(gateway3, nullptr);

    EXPECT_THROW(PaymentGatewayFactory::create("STRIPE"), InvalidArgumentException);
}
