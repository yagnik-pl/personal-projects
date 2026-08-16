#pragma once
#include <stdexcept>
#include <string>

class AppException : public std::runtime_error {
public:
    explicit AppException(const std::string& message) : std::runtime_error(message) {}
};

class NotFoundException : public AppException {
public:
    explicit NotFoundException(const std::string& message) : AppException(message) {}
};

class BadRequestException : public AppException {
public:
    explicit BadRequestException(const std::string& message) : AppException(message) {}
};

class InvalidArgumentException : public BadRequestException {
public:
    explicit InvalidArgumentException(const std::string& message) : BadRequestException(message) {}
};

class ConflictException : public AppException {
public:
    explicit ConflictException(const std::string& message) : AppException(message) {}
};

class SeatUnavailableException : public ConflictException {
public:
    explicit SeatUnavailableException(const std::string& message = "One or more requested seats are already locked or booked")
        : ConflictException(message) {}
};

class InvalidStateTransitionException : public ConflictException {
public:
    explicit InvalidStateTransitionException(const std::string& message) : ConflictException(message) {}
};

class PaymentFailedException : public BadRequestException {
public:
    explicit PaymentFailedException(const std::string& message = "Payment failed") : BadRequestException(message) {}
};
