from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import utils
from .exceptions import OTPReplayError, OTPVerificationError
from .hotp import HOTP
from .otp import OTP
from .totp import TOTP

# Bumped if the on-disk state representation ever changes incompatibly.
_STATE_VERSION = 1


def _require_non_negative_int(value: int, name: str) -> int:
    # bool is an int subclass; accepting True as 1 would silently corrupt state.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class OTPVerifier:
    """
    Stateful, replay-protected OTP verification entry point.

    The same semantics apply to :class:`~pyotp.hotp.HOTP`,
    :class:`~pyotp.totp.TOTP` and :class:`~pyotp.contrib.steam.Steam` (a TOTP
    subclass): verification scans forward over a caller-supplied window and the
    verifier remembers the highest counter position that was consumed.

    For HOTP, positions are HMAC counters in the same coordinate space as the
    ``counter`` argument of :meth:`HOTP.at` (the ``initial_count`` offset is
    handled internally). For TOTP and Steam, positions are timecodes produced by
    :meth:`TOTP.timecode`, so the configured time step and Steam's custom
    five-character alphabet take effect through the underlying OTP object.

    On a successful match the verifier:

    * returns the actual counter/timecode that matched, which may lie ahead of
      the requested base position when the client has drifted or skipped codes;
    * advances the consumed-watermark to that position.

    Any code matching at or below the watermark is rejected afterwards, even if
    it would still fall inside the requested window. Every code comparison uses
    the constant-time :func:`pyotp.utils.strings_equal` helper.

    The in-process lock makes the scan-and-advance sequence atomic between
    threads of one process. Across processes or machines, persist
    :meth:`to_state` in an external store and make the read-compare-update of
    the watermark atomic there (e.g. a conditional database update).
    """

    def __init__(self, otp: OTP, *, window: int = 0, last_consumed: int | None = None) -> None:
        """
        :param otp: a HOTP, TOTP or Steam instance used to generate candidate
            codes. A plain :class:`~pyotp.otp.OTP` base instance is rejected
            because it defines no counter coordinate space.
        :param window: default number of positions scanned forward from the
            requested base position; the base position itself is included.
        :param last_consumed: highest position already consumed, typically
            restored from :meth:`from_state`. ``None`` means no position has
            been consumed yet.
        :raises ValueError: if ``window`` or ``last_consumed`` is not a
            non-negative integer (booleans are rejected).
        :raises TypeError: if ``otp`` is not an HOTP or TOTP instance.
        """
        if not isinstance(otp, (HOTP, TOTP)):
            raise TypeError("otp must be a HOTP, TOTP or Steam instance")
        self._otp = otp
        self._window = _require_non_negative_int(window, "window")
        self._last_consumed: int | None = (
            None if last_consumed is None else _require_non_negative_int(last_consumed, "last_consumed")
        )
        self._lock = threading.Lock()

    @property
    def otp(self) -> OTP:
        """The underlying HOTP/TOTP/Steam code generator."""
        return self._otp

    @property
    def window(self) -> int:
        """Default forward window applied to :meth:`verify` calls."""
        return self._window

    @property
    def last_consumed(self) -> int | None:
        """Highest consumed position, or ``None`` before the first success."""
        return self._last_consumed

    def verify(
        self,
        otp: str,
        *,
        counter: int | None = None,
        for_time: int | float | datetime.datetime | None = None,
        window: int | None = None,
    ) -> int:
        """
        Verify ``otp`` over a forward window and atomically consume the match.

        HOTP callers must pass ``counter`` and must not pass ``for_time``;
        TOTP/Steam callers must not pass ``counter`` and may pass ``for_time``
        (defaults to the current Unix time).

        The scan covers ``[base, base + window]`` intersected with positions
        strictly above :attr:`last_consumed`, where ``base`` is the supplied
        counter (HOTP) or the timecode of ``for_time`` (TOTP/Steam). The first
        matching position in ascending order is consumed and returned.

        :param otp: the OTP submitted by the user
        :param counter: HOTP counter to scan from
        :param for_time: TOTP/Steam timestamp (Unix timestamp or datetime) to
            derive the base timecode from
        :param window: forward window for this call; defaults to the verifier's
            :attr:`window`. The upper bound is always chosen by the caller,
            never extended by the library.
        :returns: the matching position in the caller's coordinate space: the
            HOTP counter (relative to ``initial_count``) or the TOTP timecode
        :raises pyotp.exceptions.OTPReplayError: if the whole permitted range is
            at or below the consumed watermark, or the OTP only matches at or
            below it
        :raises pyotp.exceptions.OTPVerificationError: if no code within the
            permitted range matches
        :raises TypeError: on arguments that do not fit the OTP kind
        :raises ValueError: on negative or non-integer counters/windows, or an
            out-of-range timestamp
        """
        generator = self._otp
        generate: Callable[[int], str]
        if isinstance(generator, HOTP):
            if for_time is not None:
                raise TypeError("for_time is only valid for TOTP and Steam verifiers")
            if counter is None:
                raise TypeError("counter is required for HOTP verification")
            base = _require_non_negative_int(counter, "counter")
            # Keep every position (base, window, watermark, return value) in
            # the caller's counter coordinate space; generate_otp expects the
            # absolute HMAC input, so add initial_count at generation time.
            initial_count = generator.initial_count

            def generate(position: int) -> str:
                return generator.generate_otp(initial_count + position)
        else:
            if counter is not None:
                raise TypeError("counter is only valid for HOTP verification")
            if for_time is None:
                for_time = time.time()
            base = generator.timecode(for_time)
            generate = generator.generate_otp

        effective_window = self._window if window is None else _require_non_negative_int(window, "window")
        upper = base + effective_window

        with self._lock:
            lower = base if self._last_consumed is None else max(base, self._last_consumed + 1)
            if lower > upper:
                raise OTPReplayError("all positions in the verification window have already been consumed")

            candidate = str(otp)
            replay_match = False
            # If a watermark exists, codes at or below it must still be checked
            # in constant time so an actually-consumed code is reported as a
            # replay rather than a generic mismatch.
            if self._last_consumed is not None and base <= self._last_consumed:
                for position in range(base, self._last_consumed + 1):
                    if utils.strings_equal(candidate, str(generate(position))):
                        replay_match = True
                        break
            if replay_match:
                raise OTPReplayError("this OTP has already been consumed")

            for position in range(lower, upper + 1):
                if utils.strings_equal(candidate, str(generate(position))):
                    self._last_consumed = position
                    return position

        raise OTPVerificationError("OTP verification failed")

    def to_state(self) -> dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the replay state.

        The underlying OTP (including its secret and configuration) is not part
        of the state; reconstruct the same OTP object and pass it to
        :meth:`from_state` after loading. The snapshot contains only plain
        integers, strings and ``None`` and can be handed to ``json.dumps`` or
        any other serializer.
        """
        return {
            "version": _STATE_VERSION,
            "window": self._window,
            "last_consumed": self._last_consumed,
        }

    @classmethod
    def from_state(cls, otp: OTP, state: Mapping[str, Any]) -> OTPVerifier:
        """
        Reconstruct a verifier from a mapping produced by :meth:`to_state`.

        :param otp: the same HOTP/TOTP/Steam configuration the state belongs to
        :param state: mapping previously returned by :meth:`to_state`, possibly
            loaded back from JSON or another external store
        :raises ValueError: if the state is malformed, uses an unknown version
            or contains invalid window/watermark values
        :raises TypeError: if ``otp`` is not an HOTP or TOTP instance
        """
        if not isinstance(state, Mapping):
            raise ValueError("state must be a mapping")  # noqa: TRY004 - malformed serialized data, not a Python type error
        version = state.get("version")
        if version != _STATE_VERSION:
            raise ValueError(f"unsupported OTP verifier state version: {version!r}")
        if "window" not in state or "last_consumed" not in state:
            raise ValueError("state is missing required fields")
        return cls(otp, window=state["window"], last_consumed=state["last_consumed"])

    def __getstate__(self) -> dict[str, Any]:
        # Locks are not shared across processes; each unpickled copy gets a new one.
        return {"otp": self._otp, "window": self._window, "last_consumed": self._last_consumed}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._otp = state["otp"]
        self._window = state["window"]
        self._last_consumed = state["last_consumed"]
        # Locks are not shared across processes; each unpickled copy gets a new one.
        self._lock = threading.Lock()
