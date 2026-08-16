#pragma once
#include <string>
#include <memory>
#include "../../models/Enums.h"
#include "../../core/Exceptions.h"

class BookingContext;

class BookingState {
public:
    virtual ~BookingState() = default;

    virtual std::string getStatusName() const = 0;
    virtual bool canTransitionTo(const std::string& targetStatus) const = 0;

    virtual void lockSeats(BookingContext& context);
    virtual void initiatePayment(BookingContext& context);
    virtual void confirmPayment(BookingContext& context);
    virtual void cancel(BookingContext& context);
    virtual void failPayment(BookingContext& context);

    static std::shared_ptr<BookingState> fromString(const std::string& status);
};

class BookingContext {
public:
    BookingContext(int bookingId, std::shared_ptr<BookingState> initialState);
    BookingContext(int bookingId, const std::string& initialStatus);

    int getBookingId() const { return bookingId_; }
    std::shared_ptr<BookingState> getState() const { return state_; }
    std::string getStatusName() const;

    void setState(std::shared_ptr<BookingState> newState);

    void lockSeats();
    void initiatePayment();
    void confirmPayment();
    void cancel();
    void failPayment();

private:
    int bookingId_;
    std::shared_ptr<BookingState> state_;
};
