"""No DB required: password hashing and JWT are pure functions of their inputs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from fcesapi.config import get_settings
from fcesapi.security import create_access_token, hash_password, verify_password


class FakeUser:
    def __init__(self, id_: int, role: str):
        self.id = id_
        self.role = role


class TestPasswordHashing:
    def test_a_correct_password_verifies(self):
        h = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", h)

    def test_a_wrong_password_is_refused(self):
        h = hash_password("correct horse battery staple")
        assert not verify_password("wrong password", h)

    def test_the_same_password_hashes_differently_each_time(self):
        # argon2id salts per call -- two hashes of the same password must not be equal,
        # or a leaked hash table would let an attacker spot repeated passwords instantly.
        assert hash_password("x") != hash_password("x")


class TestAccessToken:
    def test_the_token_carries_the_users_id_and_role(self):
        token = create_access_token(FakeUser(7, "technician"))
        payload = jwt.decode(
            token, get_settings().jwt_secret, algorithms=[get_settings().jwt_algorithm]
        )
        assert payload["sub"] == "7"
        assert payload["role"] == "technician"

    def test_the_token_expires_in_the_future(self):
        token = create_access_token(FakeUser(1, "admin"))
        payload = jwt.decode(
            token, get_settings().jwt_secret, algorithms=[get_settings().jwt_algorithm]
        )
        assert payload["exp"] > datetime.now(UTC).timestamp()

    def test_a_token_signed_with_a_different_secret_is_rejected(self):
        token = create_access_token(FakeUser(1, "admin"))
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, "a-different-secret", algorithms=["HS256"])
