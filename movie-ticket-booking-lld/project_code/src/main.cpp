#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>
#include <crow.h>

#include "core/Database.h"
#include "repositories/UserRepository.h"
#include "repositories/CityRepository.h"
#include "repositories/TheatreRepository.h"
#include "repositories/ScreenRepository.h"
#include "repositories/MovieRepository.h"
#include "repositories/ShowRepository.h"
#include "repositories/SeatRepository.h"
#include "repositories/ShowSeatRepository.h"
#include "repositories/BookingRepository.h"
#include "repositories/PaymentRepository.h"

#include "services/payment/PaymentGatewayFactory.h"
#include "services/ShowService.h"
#include "services/BookingService.h"
#include "services/PaymentService.h"
#include "services/LockCleanupService.h"

#include "controllers/ShowController.h"
#include "controllers/BookingController.h"
#include "controllers/PaymentController.h"
#include "controllers/ResetController.h"

namespace {
std::string getEnv(const std::string& varName, const std::string& defaultValue) {
    const char* val = std::getenv(varName.c_str());
    return (val != nullptr && val[0] != '\0') ? std::string(val) : defaultValue;
}
} // anonymous namespace

int main(int argc, char* argv[]) {
    // 1. Read configuration from environment variables or defaults
    std::string dbHost = getEnv("DB_HOST", "localhost");
    std::string dbPort = getEnv("DB_PORT", "5432");
    std::string dbName = getEnv("DB_NAME", "movie_booking_db");
    std::string dbUser = getEnv("DB_USER", "postgres");
    std::string dbPassword = getEnv("DB_PASSWORD", "postgres");
    int poolSize = std::stoi(getEnv("DB_POOL_SIZE", "30"));
    int serverPort = std::stoi(getEnv("PORT", "8080"));

    std::string connStr = "host=" + dbHost +
                          " port=" + dbPort +
                          " dbname=" + dbName +
                          " user=" + dbUser +
                          " password=" + dbPassword;

    std::cout << "[INFO] Initializing Database connection pool (" << poolSize << " connections)..." << std::endl;
    std::cout << "[INFO] Connecting to " << dbHost << ":" << dbPort << "/" << dbName << " as " << dbUser << std::endl;

    std::shared_ptr<Database> db;
    try {
        db = std::make_shared<Database>(connStr, poolSize);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize database connection pool: " << e.what() << std::endl;
        return 1;
    }

    // 2. Instantiate all Repositories
    auto userRepo = std::make_shared<UserRepository>(db);
    auto cityRepo = std::make_shared<CityRepository>(db);
    auto theatreRepo = std::make_shared<TheatreRepository>(db);
    auto screenRepo = std::make_shared<ScreenRepository>(db);
    auto movieRepo = std::make_shared<MovieRepository>(db);
    auto showRepo = std::make_shared<ShowRepository>(db);
    auto seatRepo = std::make_shared<SeatRepository>(db);
    auto showSeatRepo = std::make_shared<ShowSeatRepository>(db);
    auto bookingRepo = std::make_shared<BookingRepository>(db);
    auto paymentRepo = std::make_shared<PaymentRepository>(db);

    // 3. Instantiate Payment Gateway
    auto paymentGateway = PaymentGatewayFactory::createGateway("MOCK");

    // 4. Instantiate Services
    auto showService = std::make_shared<ShowService>(showRepo, showSeatRepo);
    auto bookingService = std::make_shared<BookingService>(bookingRepo, showSeatRepo, showRepo, userRepo, 300);
    auto paymentService = std::make_shared<PaymentService>(paymentRepo, bookingRepo, showSeatRepo, paymentGateway);
    auto lockCleanupService = std::make_shared<LockCleanupService>(showSeatRepo, bookingRepo, std::chrono::milliseconds(500));

    // 5. Start Background Lock Reaper
    std::cout << "[INFO] Starting LockCleanupService background reaper thread (500ms interval)..." << std::endl;
    lockCleanupService->start();

    // 6. Instantiate Controllers
    auto showController = std::make_shared<ShowController>(showService);
    auto bookingController = std::make_shared<BookingController>(bookingService);
    auto paymentController = std::make_shared<PaymentController>(paymentService);
    auto resetController = std::make_shared<ResetController>(db);

    // 7. Setup Crow SimpleApp
    crow::SimpleApp app;

    showController->registerRoutes(app);
    bookingController->registerRoutes(app);
    paymentController->registerRoutes(app);
    resetController->registerRoutes(app);

    std::cout << "[INFO] Starting Crow HTTP Server on port " << serverPort << "..." << std::endl;
    app.port(serverPort).multithreaded().run();

    // 8. Clean up background service on shutdown
    std::cout << "[INFO] Shutting down LockCleanupService..." << std::endl;
    lockCleanupService->stop();
    std::cout << "[INFO] Server shutdown complete." << std::endl;

    return 0;
}
