"""Runtime contract/guard decorators for nifty_quant numerical functions."""
from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Literal, TypeVar, cast

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


class Strictness(IntEnum):
    OFF = 0
    CHEAP = 1
    FULL = 2


class ContractViolation(AssertionError):
    """Raised when a declared contract is broken."""


_strictness: Strictness | None = None


def get_strictness() -> Strictness:
    """Return the cached strictness level, initialising from ``NQ_STRICT`` once."""
    global _strictness
    if _strictness is not None:
        return _strictness

    raw = os.environ.get("NQ_STRICT")
    if raw in {"0", "1", "2"}:
        _strictness = Strictness(int(raw))
    else:
        _strictness = Strictness.CHEAP
    return _strictness


def set_strictness(level: Strictness) -> None:
    """Override the current strictness level."""
    global _strictness
    _strictness = level


@contextmanager
def strictness(level: Strictness) -> Iterator[None]:
    """Temporarily set strictness to *level* and restore the previous value."""
    saved = get_strictness()
    set_strictness(level)
    try:
        yield
    finally:
        set_strictness(saved)


def _bind_args(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    """Bind call arguments and apply defaults, converting bind errors to contracts."""
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
    except TypeError as exc:
        raise ContractViolation(f"{func.__name__}: could not bind arguments: {exc}") from exc
    bound.apply_defaults()
    return bound


def _resolve_arg(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key: str | int,
) -> tuple[str, Any]:
    """Resolve a named or positional argument for *func* from `args` and `kwargs`."""
    bound = _bind_args(func, args, kwargs)
    if isinstance(key, int):
        param_names = list(inspect.signature(func).parameters)
        if key < 0 or key >= len(param_names):
            raise ContractViolation(
                f"{func.__name__}: positional argument index {key} is out of range"
            )
        name = param_names[key]
    else:
        name = key

    if name not in bound.arguments:
        raise ContractViolation(f"{func.__name__}: unknown argument {key!r}")
    return name, bound.arguments[name]


def _static_arg_name(func: Callable[..., Any], key: str | int) -> str:
    """Resolve an int positional index to a parameter name once at decoration time."""
    if isinstance(key, int):
        param_names = list(inspect.signature(func).parameters)
        if key < 0 or key >= len(param_names):
            raise ContractViolation(
                f"{func.__name__}: positional argument index {key} is out of range"
            )
        return param_names[key]
    return key


def _rebind_call(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key: str | int,
    new_value: Any,
    name: str | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Replace *key* with *new_value* and return a rebuilt ``(args, kwargs)`` pair."""
    bound = _bind_args(func, args, kwargs)
    if name is None:
        name, _ = _resolve_arg(func, args, kwargs, key)
    elif name not in bound.arguments:
        raise ContractViolation(f"{func.__name__}: unknown argument {key!r}")
    bound.arguments[name] = new_value
    return bound.args, bound.kwargs


def _count_nan(value: Any) -> int:
    """Return the number of NaNs in *value*, or 0 for unsupported/non-array values."""
    if not isinstance(value, np.ndarray):
        return 0
    try:
        return int(np.isnan(value).sum())
    except (TypeError, ValueError):
        return 0


def _count_inf(value: Any) -> int:
    """Return the number of infinities in *value*, or 0 for unsupported/non-array values."""
    if not isinstance(value, np.ndarray):
        return 0
    try:
        return int(np.isinf(value).sum())
    except (TypeError, ValueError):
        return 0


def panel_contract(
    *,
    ndim: int = 2,
    dtype_out: type = np.float64,
    allow_nan: bool = True,
    allow_inf: bool = False,
    same_shape_as_arg: str | int | None = None,
    min_rows: int = 1,
) -> Callable[[F], F]:
    """Enforce ndarray shape, dtype, NaN/inf and output-shape contracts at >= CHEAP."""

    def decorator(func: F) -> F:
        same_shape_as_arg_name = (
            _static_arg_name(func, same_shape_as_arg) if same_shape_as_arg is not None else None
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level == Strictness.OFF:
                return func(*args, **kwargs)

            bound = _bind_args(func, args, kwargs)
            for name, value in bound.arguments.items():
                if isinstance(value, np.ndarray):
                    if value.ndim != ndim:
                        raise ContractViolation(
                            f"{func.__name__}: argument {name!r} must be {ndim}-D, "
                            f"got {value.ndim}-D"
                        )
                    if value.ndim > 0 and value.shape[0] < min_rows:
                        raise ContractViolation(
                            f"{func.__name__}: argument {name!r} must have at least "
                            f"{min_rows} rows, got shape {value.shape}"
                        )

            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                if result.dtype != np.dtype(dtype_out):
                    raise ContractViolation(
                        f"{func.__name__}: output dtype must be {np.dtype(dtype_out)}, "
                        f"got {result.dtype}"
                    )
                if not allow_nan:
                    nan_count = _count_nan(result)
                    if nan_count:
                        raise ContractViolation(
                            f"{func.__name__}: output contains {nan_count} NaNs"
                        )
                if not allow_inf:
                    inf_count = _count_inf(result)
                    if inf_count:
                        raise ContractViolation(
                            f"{func.__name__}: output contains {inf_count} infs"
                        )
                if same_shape_as_arg_name is not None:
                    # pragma: no cover justification — `same_shape_as_arg_name` is resolved at
                    # DECORATION time by `_static_arg_name(func, same_shape_as_arg)` against
                    # `func`'s signature, and `bound.arguments` is built at call time from that
                    # same signature. A name that resolved once therefore always binds. Kept as
                    # a guard against a future change that resolves the name from somewhere
                    # other than the signature.
                    if same_shape_as_arg_name not in bound.arguments:  # pragma: no cover
                        raise ContractViolation(
                            f"{func.__name__}: unknown argument {same_shape_as_arg!r}"
                        )
                    arg_name = same_shape_as_arg_name
                    arg_value = bound.arguments[arg_name]
                    if not isinstance(arg_value, np.ndarray):
                        raise ContractViolation(
                            f"{func.__name__}: same_shape_as_arg {arg_name!r} "
                            "must be an ndarray"
                        )
                    if result.shape != arg_value.shape:
                        raise ContractViolation(
                            f"{func.__name__}: output shape {result.shape} differs "
                            f"from {arg_name!r} shape {arg_value.shape}"
                        )
            return result

        return cast(F, wrapper)

    return decorator


def finite_output(*, allow_nan: bool = True) -> Callable[[F], F]:
    """Ensure ndarray results contain no infs (and optionally no NaNs) at >= CHEAP."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level == Strictness.OFF:
                return func(*args, **kwargs)

            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                inf_count = _count_inf(result)
                if inf_count:
                    raise ContractViolation(
                        f"{func.__name__}: output contains {inf_count} infs"
                    )
                if not allow_nan:
                    nan_count = _count_nan(result)
                    if nan_count:
                        raise ContractViolation(
                            f"{func.__name__}: output contains {nan_count} NaNs"
                        )
            return result

        return cast(F, wrapper)

    return decorator


def causal(
    *,
    row_arg: str | int = 0,
    n_probes: int = 3,
    seed: int = 0,
    domain: Literal["real", "positive"] = "real",
) -> Callable[[F], F]:
    """Probe for lookahead by perturbing future rows at FULL strictness.

    `domain` controls probe row distribution: real standard-normal or
    positive log-normal (for strictly-positive inputs).
    """

    def decorator(func: F) -> F:
        row_arg_name = _static_arg_name(func, row_arg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level < Strictness.FULL:
                return func(*args, **kwargs)

            name, x = _resolve_arg(func, args, kwargs, row_arg_name)
            if not isinstance(x, np.ndarray) or x.ndim < 1 or x.shape[0] < 2:
                raise ContractViolation(
                    f"{func.__name__}: causal row_arg {name!r} must be an ndarray "
                    "with at least two rows"
                )

            y0 = func(*args, **kwargs)
            if not isinstance(y0, np.ndarray):
                raise ContractViolation(
                    f"{func.__name__}: causal guard requires an ndarray output"
                )

            # No `n_rows < 2` early-return here: the guard above already raises
            # ContractViolation for `x.shape[0] < 2`, so n_rows >= 2 is established by this
            # point. A second check was vestigial dead code, removed rather than excluded
            # from coverage — it implied a supported "too short to probe" mode that does not
            # exist, since such inputs are rejected outright.
            n_rows = x.shape[0]

            rng = np.random.default_rng(seed)
            cut_count = min(max(n_probes, 0), n_rows - 1)
            if cut_count <= 0:
                return y0

            cut_indices = rng.choice(
                np.arange(0, n_rows - 1), size=cut_count, replace=False
            )

            if domain == "positive":
                finite_positive = x[np.isfinite(x) & (x > 0)]
                if finite_positive.size >= 1:
                    scale = float(np.median(finite_positive))
                else:
                    scale = 1.0
                if finite_positive.size >= 2:
                    sigma = float(np.std(np.log(finite_positive / scale)))
                    if not np.isfinite(sigma) or sigma <= 0:
                        sigma = 0.01
                else:
                    sigma = 0.01

            for cut in cut_indices:
                k = int(cut)
                x_perturbed = x.copy()
                replacement_shape = x[k + 1 :].shape
                draw = rng.standard_normal(size=replacement_shape)
                if domain == "positive":
                    # This must NOT be implemented as a permutation/shuffle of
                    # existing future rows. A shuffle preserves column means and
                    # would fail to catch a full-sample-mean leak (e.g.
                    # x - np.nanmean(x, axis=0)). Drawing fresh log-normal values
                    # changes the aggregate statistics of the future rows.
                    x_perturbed[k + 1 :] = (
                        scale * np.exp(draw * sigma)
                    ).astype(x.dtype)
                else:
                    x_perturbed[k + 1 :] = draw.astype(x.dtype)

                new_args, new_kwargs = _rebind_call(
                    func, args, kwargs, row_arg, x_perturbed, row_arg_name
                )
                y1 = func(*new_args, **new_kwargs)

                if not isinstance(y1, np.ndarray) or y1.shape != y0.shape:
                    raise ContractViolation(
                        f"{func.__name__}: causal probe at cut {k} returned "
                        f"shape {getattr(y1, 'shape', None)} but baseline shape "
                        f"was {y0.shape}"
                    )

                if not np.array_equal(y0[: k + 1], y1[: k + 1], equal_nan=True):
                    for row_index in range(k + 1):
                        if not np.array_equal(
                            y0[row_index], y1[row_index], equal_nan=True
                        ):
                            raise ContractViolation(
                                f"{func.__name__}: causal violation at cut {k}, "
                                f"first mismatching row {row_index}"
                            )
                    # pragma: no cover justification — unreachable fallback. The enclosing
                    # `array_equal(y0[:k+1], y1[:k+1])` returned False, so at least one row in
                    # that slice must differ, and the loop above scans every row in it and
                    # raises on the first mismatch. Reaching here would mean a slice compares
                    # unequal while every one of its rows compares equal. Kept so the failure
                    # is a named ContractViolation rather than a silent fall-through if the
                    # comparison strategy above ever changes.
                    raise ContractViolation(  # pragma: no cover
                        f"{func.__name__}: causal violation at cut {k} "
                        "but row loop found no mismatch"
                    )
            return y0

        return cast(F, wrapper)

    return decorator


def _source_snippet(func: Callable[..., Any]) -> str:
    """Best-effort short source snippet for violation messages.

    `func.__name__` after `functools.wraps` is usually the thin wrapper's own name
    (e.g. `"guarded"`), which tells a reader nothing about what leaked. Recovering
    the wrapper's literal source (its `def` line plus body, and any decorator lines
    above it) surfaces the actual call it made, at essentially no cost: this is
    best-effort only, silently empty for functions with no retrievable source
    (builtins, exec'd code, C extensions).
    """
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return ""
    return "\nsource: " + src.strip()[:300]


def universe_causal(
    *,
    panel_arg: str | int = 0,
    row_arg: str | int | None = None,
    n_probes: int = 4,
    seed: int = 0,
) -> Callable[[F], F]:
    """Probe for universe-causality leaks by perturbing panel data after session `d`.

    Wraps `f(panel) -> mask` where `panel` is 2-D `(n_sessions, n_symbols)`, either
    bool (`True` = present) or float (`NaN` = absent, the repo-wide convention).
    `row_arg` is an alias for `panel_arg` (kept for a second caller convention); if
    given it wins.

    At FULL strictness only: perturbs a SAMPLE of sessions (always the first, the
    last, and one session adjacent to a symbol's first-present boundary if one
    exists, topped up to `n_probes` distinct sessions via a seeded, pinnable draw --
    not every session, so this stays usable on hundreds of sessions). For each
    probed session `d`, BOTH directions are checked in separate calls: making
    currently-absent future cells appear, and making currently-present future cells
    disappear. A one-directional guard passes a mask built from
    `np.any(present, axis=0)` -- see the module docstring / spec for why both
    directions are mandatory.

    Blind spot (documented per spec section D, style of
    `research/expectancy.py:939`): this guard can only prove eligibility at session
    `d` is unaffected by panel cells strictly AFTER `d`. It cannot prove eligibility
    at `d` was honestly computed from cells <= `d` -- e.g. a caller who centred a
    liquidity feature on a forward-looking window before ever passing it into the
    eligibility function has already baked the leak into session `d`'s own input,
    and this guard has nothing to say about it, exactly as `@causal` cannot see a
    leak already baked into `prior_adv` before it reaches `expectancy_by_liquidity`.
    """

    def decorator(func: F) -> F:
        key = row_arg if row_arg is not None else panel_arg
        panel_arg_name = _static_arg_name(func, key)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level < Strictness.FULL:
                return func(*args, **kwargs)

            name, panel = _resolve_arg(func, args, kwargs, panel_arg_name)
            if not isinstance(panel, np.ndarray) or panel.ndim != 2:
                raise ContractViolation(
                    f"{func.__name__}: universe_causal argument {name!r} must be a "
                    f"2-D ndarray, got {type(panel).__name__}"
                )

            y0 = func(*args, **kwargs)
            if not isinstance(y0, np.ndarray) or y0.shape[0] != panel.shape[0]:
                raise ContractViolation(
                    f"{func.__name__}: universe_causal output must be an ndarray "
                    f"with shape[0] == {panel.shape[0]}, got {type(y0).__name__}"
                )

            n_sessions = panel.shape[0]
            if n_sessions < 2:
                return y0

            present = panel if panel.dtype == np.bool_ else np.isfinite(panel)

            # Always probe the first and last session, plus one session adjacent to
            # a symbol's first-present boundary if one exists.
            probe_set: set[int] = {0, n_sessions - 1}
            fp_candidates: list[int] = []
            for col in range(present.shape[1]):
                true_indices = np.nonzero(present[:, col])[0]
                if true_indices.size > 0 and true_indices[0] > 0:
                    fp_candidates.append(int(true_indices[0]))
            if fp_candidates:
                probe_set.add(min(fp_candidates) - 1)

            # Top up to n_probes distinct sessions with a seeded, pinnable draw --
            # this is a SAMPLE, not every session, so FULL strictness stays usable.
            rng = np.random.default_rng(seed)
            remaining = [i for i in range(0, n_sessions - 1) if i not in probe_set]
            budget = max(n_probes - len(probe_set), 0)
            if budget > 0 and remaining:
                n_draw = min(budget, len(remaining))
                drawn = rng.choice(remaining, size=n_draw, replace=False)
                probe_set.update(int(d) for d in drawn)

            for d in sorted(probe_set):
                # Add direction: currently-absent future cells become present.
                panel_add = panel.copy()
                future_absent = ~present[d + 1 :]
                if panel.dtype == np.bool_:
                    panel_add[d + 1 :][future_absent] = True
                else:
                    draw = rng.standard_normal(size=int(future_absent.sum()))
                    panel_add[d + 1 :][future_absent] = draw.astype(panel.dtype)

                new_args, new_kwargs = _rebind_call(
                    func, args, kwargs, key, panel_add, panel_arg_name
                )
                y1 = func(*new_args, **new_kwargs)
                if not isinstance(y1, np.ndarray) or y1.shape[0] != y0.shape[0]:
                    raise ContractViolation(
                        f"{func.__name__}: universe_causal add-direction probe at "
                        f"session {d} returned shape "
                        f"{getattr(y1, 'shape', None)}, baseline was {y0.shape}"
                    )
                # array_equal(equal_nan=True) treats matching NaN patterns as equal
                # and correctly flags a NaN-vs-finite mismatch, unlike a magnitude
                # check `abs(x - y) > atol`, which is silently False (no violation)
                # whenever either side is NaN.
                if not np.array_equal(y1[d], y0[d], equal_nan=True):
                    raise ContractViolation(
                        f"{func.__name__}: universe_causal violation (add "
                        f"direction): eligibility mask row/session {d} changed when "
                        f"absent symbols were made to appear strictly after session "
                        f"{d}{_source_snippet(func)}"
                    )

                # Remove direction: currently-present future cells become absent.
                panel_remove = panel.copy()
                future_present = present[d + 1 :]
                if panel.dtype == np.bool_:
                    panel_remove[d + 1 :][future_present] = False
                else:
                    panel_remove[d + 1 :][future_present] = np.nan

                new_args, new_kwargs = _rebind_call(
                    func, args, kwargs, key, panel_remove, panel_arg_name
                )
                y1 = func(*new_args, **new_kwargs)
                if not isinstance(y1, np.ndarray) or y1.shape[0] != y0.shape[0]:
                    raise ContractViolation(
                        f"{func.__name__}: universe_causal remove-direction probe "
                        f"at session {d} returned shape "
                        f"{getattr(y1, 'shape', None)}, baseline was {y0.shape}"
                    )
                if not np.array_equal(y1[d], y0[d], equal_nan=True):
                    raise ContractViolation(
                        f"{func.__name__}: universe_causal violation (remove "
                        f"direction): eligibility mask row/session {d} changed when "
                        f"present symbols were made to disappear strictly after "
                        f"session {d}{_source_snippet(func)}"
                    )

            return y0

        return cast(F, wrapper)

    return decorator


def execution_causal(
    *,
    row_arg: str | int | None = None,
    row_args: Sequence[str | int] | None = None,
    n_probes: int = 3,
    seed: int = 0,
) -> Callable[[F], F]:
    """Probe for execution-causality leaks by perturbing fill inputs after row `t`.

    Two calling conventions, both real (two independent test suites used both, and
    the signature below lets either pass -- see module-level rationale):

    - `row_arg`: a single 2-D ndarray argument (e.g. packed columns
      `[orders, price, bar_traded_value, tradable]`); the whole future block is
      redrawn at once, exactly like `@causal(domain="real")`.
    - `row_args`: a sequence of 1-D ndarray argument names/indices (e.g.
      `("prices", "bar_traded_value")`) on a function taking several separate
      arguments; only the named arguments are perturbed (with positive log-normal
      noise, matching `@causal(domain="positive")`) -- other arguments such as
      `orders`/`tradable` are decisions already known at `t`, not leakable market
      data, and are passed through unchanged.

    At FULL strictness only: perturbs a SAMPLE of rows (row 0 always, topped up to
    `n_probes` distinct rows via a seeded, pinnable draw -- not every row, so FULL
    strictness stays usable on large books). Specifically exercises the
    participation cap (`fills.py:101`, `max_participation * bar_traded_value`)
    because `bar_traded_value` is one of the perturbed inputs.

    Blind spot (documented per spec section D, mirroring
    `research/expectancy.py:939`): this guard can only prove row `t`'s fill output
    is unaffected by rows strictly AFTER `t`. It cannot prove row `t`'s OWN
    `bar_traded_value`/`prices` were honestly computed from data <= `t` before ever
    reaching the wrapped fill function -- e.g. a caller who centred
    `bar_traded_value` on a forward-looking window has already contaminated row
    `t`'s own input, and this guard raises nothing, exactly as `@causal` cannot see
    a leak already baked into `prior_adv` before it reaches
    `expectancy_by_liquidity`.
    """

    def decorator(func: F) -> F:
        if row_arg is not None and row_args is not None:
            raise ValueError(
                "execution_causal: specify exactly one of row_arg or row_args"
            )
        # F13: an EMPTY row_args silently disabled the guard entirely, at every
        # strictness level including FULL. `static_names` came out empty, so the
        # validation loop never ran, `n_rows` was never assigned, and the
        # `n_rows is None` early return called straight through to the wrapped
        # function with ZERO checking -- a deliberately leaky function passed.
        # Fail at DECORATION time: an empty container is not "nothing to check",
        # it is a caller error. Same family as F12's `if col_idx:`.
        if row_args is not None and len(row_args) == 0:
            raise ValueError(
                "execution_causal: row_args is empty; that would silently disable "
                "the guard. Pass the argument names to perturb, or omit row_args "
                "to use the default single-array convention."
            )
        if row_arg is None and row_args is None:
            effective_row_arg: str | int | None = 0
            effective_row_args = None
        else:
            effective_row_arg = row_arg
            effective_row_args = row_args

        if effective_row_arg is not None:
            row_arg_name = _static_arg_name(func, effective_row_arg)

            @functools.wraps(func)
            def _wrapper_row_arg(*args: Any, **kwargs: Any) -> Any:
                _level = _strictness
                if _level is None:
                    _level = get_strictness()
                if _level < Strictness.FULL:
                    return func(*args, **kwargs)

                name, x = _resolve_arg(func, args, kwargs, row_arg_name)
                if not isinstance(x, np.ndarray) or x.ndim != 2:
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal argument {name!r} must "
                        f"be a 2-D ndarray, got {type(x).__name__}"
                    )

                y0 = func(*args, **kwargs)
                if not isinstance(y0, np.ndarray) or y0.shape[0] != x.shape[0]:
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal output must be an "
                        f"ndarray with shape[0] == {x.shape[0]}, got "
                        f"{type(y0).__name__}"
                    )

                n_rows = x.shape[0]
                if n_rows < 2:
                    return y0

                rng = np.random.default_rng(seed)
                probe_set: set[int] = {0}
                remaining = list(range(1, n_rows - 1))
                budget = min(n_probes, n_rows - 1) - 1
                if budget > 0 and remaining:
                    n_draw = min(budget, len(remaining))
                    drawn = rng.choice(remaining, size=n_draw, replace=False)
                    probe_set.update(int(d) for d in drawn)

                for t in sorted(probe_set):
                    x_perturbed = x.copy()
                    draw = rng.standard_normal(size=x[t + 1 :].shape)
                    x_perturbed[t + 1 :] = draw.astype(x.dtype)

                    new_args, new_kwargs = _rebind_call(
                        func, args, kwargs, effective_row_arg, x_perturbed,
                        row_arg_name,
                    )
                    y1 = func(*new_args, **new_kwargs)
                    if not isinstance(y1, np.ndarray) or y1.shape != y0.shape:
                        raise ContractViolation(
                            f"{func.__name__}: execution_causal probe at row {t} "
                            f"returned shape {getattr(y1, 'shape', None)}, "
                            f"baseline was {y0.shape}"
                        )
                    # See universe_causal for why equal_nan=True, not a magnitude
                    # threshold, is used here.
                    if not np.array_equal(y1[t], y0[t], equal_nan=True):
                        raise ContractViolation(
                            f"{func.__name__}: execution_causal violation: fill "
                            f"quantity/price/charges at row {t} changed when data "
                            f"strictly after row {t} was perturbed"
                            f"{_source_snippet(func)}"
                        )

                return y0

            return cast(F, _wrapper_row_arg)

        static_names = [_static_arg_name(func, k) for k in effective_row_args]  # type: ignore[union-attr]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level < Strictness.FULL:
                return func(*args, **kwargs)

            resolved: list[np.ndarray] = []
            n_rows: int | None = None
            for arg_name in static_names:
                _, val = _resolve_arg(func, args, kwargs, arg_name)
                if not isinstance(val, np.ndarray) or val.ndim != 1:
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal row_args entry "
                        f"{arg_name!r} must be a 1-D ndarray, got "
                        f"{type(val).__name__}"
                    )
                if n_rows is None:
                    n_rows = val.shape[0]
                elif val.shape[0] != n_rows:
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal row_args entries must "
                        f"share one length, got {n_rows} and {val.shape[0]} for "
                        f"{arg_name!r}"
                    )
                resolved.append(val)

            if n_rows is None or n_rows < 2:
                return func(*args, **kwargs)

            y0 = func(*args, **kwargs)
            if not isinstance(y0, np.ndarray) or y0.shape[0] != n_rows:
                raise ContractViolation(
                    f"{func.__name__}: execution_causal output must be an ndarray "
                    f"with shape[0] == {n_rows}, got {type(y0).__name__}"
                )

            rng = np.random.default_rng(seed)
            probe_set = {0}
            remaining = list(range(1, n_rows - 1))
            budget = min(n_probes, n_rows - 1) - 1
            if budget > 0 and remaining:
                n_draw = min(budget, len(remaining))
                drawn = rng.choice(remaining, size=n_draw, replace=False)
                probe_set.update(int(d) for d in drawn)

            for t in sorted(probe_set):
                perturbed: list[np.ndarray] = []
                for arr in resolved:
                    arr_copy = arr.copy()
                    finite_positive = arr[np.isfinite(arr) & (arr > 0)]
                    scale = (
                        float(np.median(finite_positive))
                        if finite_positive.size >= 1
                        else 1.0
                    )
                    if finite_positive.size >= 2:
                        sigma = float(np.std(np.log(finite_positive / scale)))
                        if not np.isfinite(sigma) or sigma <= 0:
                            sigma = 0.01
                    else:
                        sigma = 0.01
                    draw = rng.standard_normal(size=arr[t + 1 :].shape)
                    arr_copy[t + 1 :] = (scale * np.exp(draw * sigma)).astype(
                        arr.dtype
                    )
                    # Row t+1 specifically gets a forced, deterministic, extreme
                    # low value on top of the general jitter above: this is the
                    # exact off-by-one leak the spec calls out (fills.py:101 --
                    # "bar_traded_value is the one input where reading the wrong
                    # row is both easy and invisible"). A participation cap only
                    # binds, and so only becomes visible in the output, when the
                    # future value is small enough relative to the order size; a
                    # mild same-order-of-magnitude jitter of the whole future
                    # block (as used for rows after t+1) can easily miss that
                    # threshold by luck of the seed for realistic order sizes, so
                    # row t+1 is forced deterministically rather than left to
                    # probability.
                    arr_copy[t + 1] = arr.dtype.type(scale * 1e-6)
                    perturbed.append(arr_copy)

                new_args, new_kwargs = args, kwargs
                for arg_name, new_val in zip(static_names, perturbed):
                    new_args, new_kwargs = _rebind_call(
                        func, new_args, new_kwargs, arg_name, new_val, arg_name
                    )

                y1 = func(*new_args, **new_kwargs)
                if not isinstance(y1, np.ndarray) or y1.shape != y0.shape:
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal probe at row {t} "
                        f"returned shape {getattr(y1, 'shape', None)}, baseline "
                        f"was {y0.shape}"
                    )
                if not np.array_equal(y1[t], y0[t], equal_nan=True):
                    raise ContractViolation(
                        f"{func.__name__}: execution_causal violation: fill "
                        f"quantity/price/charges at row {t} changed when "
                        f"{', '.join(repr(n) for n in static_names)} strictly "
                        f"after row {t} were perturbed{_source_snippet(func)}"
                    )

            return y0

        return cast(F, wrapper)

    return decorator


def deterministic(*, n_repeats: int = 2) -> Callable[[F], F]:
    """Check that repeated FULL-strictness calls produce identical results."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level < Strictness.FULL:
                return func(*args, **kwargs)

            first = func(*args, **kwargs)
            for repeat in range(2, max(n_repeats, 1) + 1):
                current = func(*args, **kwargs)
                if isinstance(first, np.ndarray) and isinstance(current, np.ndarray):
                    same = bool(np.array_equal(first, current, equal_nan=True))
                elif not isinstance(first, np.ndarray) and not isinstance(
                    current, np.ndarray
                ):
                    same = bool(first == current)
                else:
                    same = False
                if not same:
                    raise ContractViolation(
                        f"{func.__name__}: non-deterministic result at repeat {repeat}"
                    )
            return first

        return cast(F, wrapper)

    return decorator


def monotone_in(
    arg: str | int,
    *,
    increasing: bool = True,
    n_probes: int = 5,
    seed: int = 0,
) -> Callable[[F], F]:
    """Probe FULL-strictness monotonicity of a scalar output in one input argument."""

    def decorator(func: F) -> F:
        arg_name = _static_arg_name(func, arg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level < Strictness.FULL:
                return func(*args, **kwargs)

            name, current_value = _resolve_arg(func, args, kwargs, arg_name)
            if not isinstance(current_value, (int, float, np.integer, np.floating)):
                raise ContractViolation(
                    f"{func.__name__}: monotone_in argument {name!r} must be "
                    "an int or float scalar"
                )

            y0 = func(*args, **kwargs)
            try:
                base_output = float(y0)
            except (TypeError, ValueError) as exc:
                raise ContractViolation(
                    f"{func.__name__}: monotone_in output must be float-coercible"
                ) from exc

            base = float(current_value)
            rng = np.random.default_rng(seed)
            probe_count = max(n_probes, 0)
            offsets = rng.uniform(0.01, 5.0, size=probe_count)
            values = sorted({base} | {base + float(offset) for offset in offsets})

            outputs: list[float] = []
            for v in values:
                if v == base:
                    outputs.append(base_output)
                else:
                    new_args, new_kwargs = _rebind_call(func, args, kwargs, arg, v, arg_name)
                    candidate = func(*new_args, **new_kwargs)
                    try:
                        outputs.append(float(candidate))
                    except (TypeError, ValueError) as exc:
                        raise ContractViolation(
                            f"{func.__name__}: monotone_in probe output must be "
                            "float-coercible"
                        ) from exc

            for i in range(len(values) - 1):
                if increasing and outputs[i + 1] < outputs[i] - 1e-9:
                    raise ContractViolation(
                        f"{func.__name__}: monotone_in violation on {name!r}: "
                        f"outputs {outputs[i]} at arg value {values[i]} and "
                        f"{outputs[i + 1]} at {values[i + 1]} are not non-decreasing"
                    )
                if not increasing and outputs[i + 1] > outputs[i] + 1e-9:
                    raise ContractViolation(
                        f"{func.__name__}: monotone_in violation on {name!r}: "
                        f"outputs {outputs[i]} at arg value {values[i]} and "
                        f"{outputs[i + 1]} at {values[i + 1]} are not non-increasing"
                    )

            return y0

        return cast(F, wrapper)

    return decorator


def bounded_output(
    *,
    lo: float | None = None,
    hi: float | None = None,
    inclusive: bool = True,
) -> Callable[[F], F]:
    """Ensure finite scalar/array outputs lie within specified bounds at >= CHEAP."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level == Strictness.OFF:
                return func(*args, **kwargs)

            result = func(*args, **kwargs)
            arr = np.asarray(result)
            try:
                finite_mask = np.isfinite(arr)
            except TypeError:
                return result

            finite_vals = arr[finite_mask]

            if lo is not None:
                if inclusive:
                    bad = finite_vals < lo
                else:
                    bad = finite_vals <= lo
                bad_count = int(bad.sum())
                if bad_count:
                    raise ContractViolation(
                        f"{func.__name__}: {bad_count} finite output values "
                        f"violate lower bound {lo}"
                    )

            if hi is not None:
                if inclusive:
                    bad = finite_vals > hi
                else:
                    bad = finite_vals >= hi
                bad_count = int(bad.sum())
                if bad_count:
                    raise ContractViolation(
                        f"{func.__name__}: {bad_count} finite output values "
                        f"violate upper bound {hi}"
                    )

            return result

        return cast(F, wrapper)

    return decorator


def no_forward_fill(*, arg: str | int = 0) -> Callable[[F], F]:
    """Reject functions that silently reduce the number of NaNs in *arg* at >= CHEAP."""

    def decorator(func: F) -> F:
        arg_name = _static_arg_name(func, arg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level == Strictness.OFF:
                return func(*args, **kwargs)

            name, arr = _resolve_arg(func, args, kwargs, arg_name)
            if not isinstance(arr, np.ndarray):
                raise ContractViolation(
                    f"{func.__name__}: no_forward_fill argument {name!r} "
                    "must be an ndarray"
                )

            nan_before = _count_nan(arr)
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                nan_after = _count_nan(result)
            else:
                nan_after = 0

            if nan_after < nan_before:
                raise ContractViolation(
                    f"{func.__name__}: forward-fill repair detected for {name!r}: "
                    f"NaN count fell from {nan_before} to {nan_after}"
                )
            return result

        return cast(F, wrapper)

    return decorator


def validate_weights(*, max_gross: float = 1.0, tol: float = 1e-9) -> Callable[[F], F]:
    """Validate 1-D weight vectors for finiteness and gross exposure at >= CHEAP."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _level = _strictness
            if _level is None:
                _level = get_strictness()
            if _level == Strictness.OFF:
                return func(*args, **kwargs)

            result = func(*args, **kwargs)
            w = np.asarray(result)
            if w.ndim != 1:
                raise ContractViolation(
                    f"{func.__name__}: weight output must be 1-D, got shape {w.shape}"
                )

            finite_mask = np.isfinite(w)
            non_finite_count = int((~finite_mask).sum())
            if non_finite_count:
                raise ContractViolation(
                    f"{func.__name__}: weight output contains "
                    f"{non_finite_count} non-finite elements"
                )

            gross = float(np.sum(np.abs(w)))
            if gross > max_gross * (1 + tol):
                raise ContractViolation(
                    f"{func.__name__}: gross exposure {gross} exceeds "
                    f"max_gross {max_gross}"
                )
            return w

        return cast(F, wrapper)

    return decorator


def check(
    condition: bool,
    msg: str,
    *,
    level: Strictness = Strictness.CHEAP,
) -> None:
    """Raise ``ContractViolation`` when *condition* is false and strictness allows.

    The boolean condition is evaluated by the caller before this function is
    entered; that is inherent to this being a plain function rather than a macro.
    """
    _level = _strictness
    if _level is None:
        _level = get_strictness()
    if _level >= level and not condition:
        raise ContractViolation(msg)


def check_accounting(
    cash: float,
    positions_value: float,
    costs: float,
    initial_capital: float,
    pnl: float,
    *,
    atol: float = 1e-6,
) -> None:
    """Check accounting identity at FULL strictness."""
    _level = _strictness
    if _level is None:
        _level = get_strictness()
    if _level < Strictness.FULL:
        return

    discrepancy = (cash + positions_value + costs) - (initial_capital + pnl)
    if not np.isfinite(discrepancy) or abs(discrepancy) > atol:
        raise ContractViolation(
            f"accounting check failed: cash={cash}, positions_value={positions_value}, "
            f"costs={costs}, initial_capital={initial_capital}, pnl={pnl}, "
            f"discrepancy={discrepancy}"
        )


def check_aligned(*arrays: np.ndarray, names: Sequence[str] | None = None) -> None:
    """Check that all supplied arrays have the same leading length, always."""
    if len(arrays) < 2:
        return

    shapes = [a.shape[0] for a in arrays]
    for i in range(1, len(shapes)):
        if shapes[i] != shapes[0]:
            name0 = names[0] if names is not None and len(names) > 0 else "arg0"
            name_i = names[i] if names is not None and len(names) > i else f"arg{i}"
            raise ContractViolation(
                f"alignment check failed: {name0} has length {shapes[0]}, "
                f"{name_i} has length {shapes[i]}"
            )


def check_no_all_nan_columns(
    x: np.ndarray,
    names: Sequence[str] | None = None,
) -> None:
    """Ensure a 2-D array has no fully-NaN columns, always."""
    if x.ndim != 2:
        raise ContractViolation(
            f"check_no_all_nan_columns: expected 2-D array, got ndim={x.ndim}"
        )

    offenders: list[int] = []
    for j in range(x.shape[1]):
        column = x[:, j]
        try:
            all_nan = bool(np.all(np.isnan(column)))
        except (TypeError, ValueError):
            all_nan = False
        if all_nan:
            offenders.append(j)

    if offenders:
        details: list[str] = []
        for j in offenders:
            if names is not None and len(names) > j:
                details.append(f"{j} ({names[j]})")
            else:
                details.append(str(j))
        raise ContractViolation(
            "check_no_all_nan_columns: all-NaN columns: " + ", ".join(details)
        )


def check_sorted_unique(ts: np.ndarray) -> None:
    """Check that a 1-D array is non-empty and strictly increasing, always."""
    if ts.ndim != 1:
        raise ContractViolation(
            f"check_sorted_unique: expected 1-D array, got ndim={ts.ndim}"
        )
    if len(ts) < 1:
        raise ContractViolation("check_sorted_unique: array must not be empty")
    if len(ts) < 2:
        return

    diffs = np.diff(ts)
    for i, diff in enumerate(diffs):
        if diff <= 0:
            raise ContractViolation(
                f"check_sorted_unique: values not strictly increasing at index {i}: "
                f"{ts[i]} followed by {ts[i + 1]}"
            )


def check_day_offsets(day_offsets: np.ndarray, n_rows: int) -> None:
    """Validate session day-offset boundaries.

    This guards the "never assume 375 bars" rule: session lengths vary
    (Muhurat sessions near 60 bars, special/DR sessions near 105 bars).
    """
    if day_offsets.ndim != 1:
        raise ContractViolation(
            f"check_day_offsets: expected 1-D array, got ndim={day_offsets.ndim}"
        )
    if len(day_offsets) < 1:
        raise ContractViolation("check_day_offsets: day_offsets must not be empty")

    if day_offsets[0] != 0:
        raise ContractViolation(
            f"check_day_offsets: first day offset must be 0, got {day_offsets[0]}"
        )
    if day_offsets[-1] != n_rows:
        raise ContractViolation(
            f"check_day_offsets: last day offset must equal n_rows={n_rows}, "
            f"got {day_offsets[-1]}"
        )

    diffs = np.diff(day_offsets)
    for i, diff in enumerate(diffs):
        if diff <= 0:
            raise ContractViolation(
                f"check_day_offsets: day offsets not strictly increasing at index {i}: "
                f"{day_offsets[i]} then {day_offsets[i + 1]}"
            )
