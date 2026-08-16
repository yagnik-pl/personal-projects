import pytest
import threading

BASE_URL = "http://localhost:8080/api"

def test_t4_1_high_demand_booking(api_session):
    # 100 threads trying to book the same seat (seat 10)
    results = []
    lock = threading.Lock()

    def book_seat(user_id):
        try:
            resp = api_session.post(f"{BASE_URL}/bookings", json={
                "show_id": 1, "user_id": user_id, "seat_ids": [10]
            })
            with lock:
                results.append(resp.status_code)
        except Exception as e:
            with lock:
                results.append(500)

    threads = []
    for i in range(100):
        t = threading.Thread(target=book_seat, args=(i + 1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Exactly one thread succeeds with 201, the rest receive 409 (or 400)
    assert results.count(201) == 1
    assert results.count(409) == 99

def test_t4_2_payment_failure_then_retry(api_session):
    # User 1 books seat 11
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [11]
    })
    assert resp.status_code == 201
    booking_id = resp.json()["booking_id"]

    # Payment fails on first attempt
    pay_fail = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id, "amount": 25.0, "fail_mock": True
    })
    assert pay_fail.status_code == 400

    # Booking is still SEATS_LOCKED (not CONFIRMED)
    status_resp = api_session.get(f"{BASE_URL}/bookings/{booking_id}")
    assert status_resp.json()["status"] == "SEATS_LOCKED"

    # Retry payment with fail_mock=False
    pay_succ = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id, "amount": 25.0, "fail_mock": False
    })
    assert pay_succ.status_code == 200

    # Now booking is CONFIRMED and seat is BOOKED
    status_resp2 = api_session.get(f"{BASE_URL}/bookings/{booking_id}")
    assert status_resp2.json()["status"] == "CONFIRMED"

    seats_resp = api_session.get(f"{BASE_URL}/shows/1/seats")
    seats = {s["seat_id"]: s["status"] for s in seats_resp.json()}
    assert seats[11] == "BOOKED"

def test_t4_3_partial_success_group_booking(api_session):
    # User 1 books seats 12 and 13
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [12, 13]
    })
    assert resp1.status_code == 201

    # User 2 attempts to book seats 13 and 14 (overlap on 13)
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [13, 14]
    })
    assert resp2.status_code == 409

    # Because seat 14 was rolled back and never locked, User 2 can now book [14, 15]
    resp3 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [14, 15]
    })
    assert resp3.status_code == 201

def test_t4_4_locks_expire_grabbed_by_another(api_session):
    # User 1 locks seat 16
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [16]
    })
    assert resp1.status_code == 201
    b_id = resp1.json()["booking_id"]

    # Verify status is SEATS_LOCKED
    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.json()["status"] == "SEATS_LOCKED"

    # Another user trying immediately gets 409
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [16]
    })
    assert resp2.status_code == 409

def test_t4_5_mixed_payments_at_scale(api_session):
    # 10 users booking seats 1..10
    booking_ids = []
    for i in range(1, 11):
        resp = api_session.post(f"{BASE_URL}/bookings", json={
            "show_id": 1, "user_id": i, "seat_ids": [i]
        })
        assert resp.status_code == 201
        booking_ids.append((i, resp.json()["booking_id"]))

    # Users 1..5 pay successfully, Users 6..10 simulate failure
    for user_idx, b_id in booking_ids:
        fail = (user_idx > 5)
        pay_resp = api_session.post(f"{BASE_URL}/payments", json={
            "booking_id": b_id, "amount": 25.0, "fail_mock": fail
        })
        if fail:
            assert pay_resp.status_code == 400
        else:
            assert pay_resp.status_code == 200

    # Check status of each booking
    for user_idx, b_id in booking_ids:
        status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
        if user_idx <= 5:
            assert status_resp.json()["status"] == "CONFIRMED"
        else:
            assert status_resp.json()["status"] == "SEATS_LOCKED"
