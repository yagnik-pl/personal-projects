# API Contract

Base URL: `http://localhost:8080`

## Endpoints

### 1. Shows & Seats
- `GET /api/shows`
  - Returns a list of available shows.
- `GET /api/shows/{show_id}/seats`
  - Returns all seats for a specific show and their current availability status (AVAILABLE, LOCKED, BOOKED).

### 2. Bookings
- `POST /api/bookings`
  - Creates a new booking.
  - Request Body:
    ```json
    {
      "show_id": 1,
      "user_id": 101,
      "seat_ids": [10, 11]
    }
    ```
  - Response: `201 Created` with `{"booking_id": 1001, "status": "SEATS_LOCKED", "total_price": 25.0}`
  - If seats are already locked/booked, returns `409 Conflict`.

- `GET /api/bookings/{booking_id}`
  - Retrieves the status of a booking.
  - Response: `{"booking_id": 1001, "status": "CONFIRMED"}` (Status could be CREATED, SEATS_LOCKED, PAYMENT_PENDING, CONFIRMED, CANCELLED)

### 3. Payments
- `POST /api/payments`
  - Submits a payment for a booking.
  - Request Body:
    ```json
    {
      "booking_id": 1001,
      "amount": 25.0,
      "fail_mock": false 
    }
    ```
  - `fail_mock` is used to simulate a payment failure for testing purposes.
  - Response: `200 OK` with `{"payment_id": 5001, "status": "SUCCESS"}` or `400 Bad Request` if `fail_mock` is true.

### 4. Admin/Testing Utilities
- `POST /api/test/reset`
  - Clears/resets database state for testing purposes.
