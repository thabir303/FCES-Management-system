"""Auth: argon2id password hashing, JWT issue/verify, role enforcement (§7.2, §12.3).

`require_role("technician")` allows technician **and** admin -- role is a floor, not an
exact match. `readonly` issuing a write gets 403 (authenticated, wrong role), never 401
(that is reserved for a missing or invalid token).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from fcesapi.config import get_settings
from fcesapi.db import get_db
from fcesapi.models import User, UserRole

_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)

#: A floor role admits itself and every role above it. `technician` allows technician and
#: admin; `readonly` allows all three. Order matters and is the whole implementation.
_ROLE_ORDER = [UserRole.readonly, UserRole.technician, UserRole.admin]


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_access_token(user: User) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "role": user.role.value if isinstance(user.role, UserRole) else user.role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "not_authenticated", "message": detail},
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None:
        raise _unauthorized("no bearer token supplied")
    settings = get_settings()
    try:
        payload = jwt.decode(
            credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized("token invalid or expired") from exc

    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise _unauthorized("user not found or inactive")
    return user


def require_role(minimum: UserRole):
    """A dependency factory: `Depends(require_role(UserRole.technician))`.

    Wrong role -> 403 (authenticated-but-forbidden), never 401. Confusing the two would
    tell an attacker that a valid-looking token merely lacks permission versus is outright
    invalid, but more importantly it violates the error envelope contract (§12.2) this
    project holds everywhere else: the two codes mean different things and callers rely on
    the distinction.
    """
    floor = _ROLE_ORDER.index(minimum)

    def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        role = user.role if isinstance(user.role, UserRole) else UserRole(user.role)
        if _ROLE_ORDER.index(role) < floor:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "forbidden",
                    "message": f"requires role '{minimum.value}' or higher",
                },
            )
        return user

    return dependency
