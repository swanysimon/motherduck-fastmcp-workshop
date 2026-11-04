"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import median
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import caller_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sf-311")

mcp = FastMCP("sf-311")


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

GroupBy = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroupBy = Literal["neighborhood", "agency"]
GROUPS: dict[str, Callable[[Request], str]] = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}


def _check(name: str | None, choices: list[str], kind: str) -> None:
    if name is not None and name not in choices:
        suggestions = ", ".join(closest(name, choices)) or "none close"
        raise ToolError(f"Unknown {kind} {name!r}. Did you mean: {suggestions}? Call dataset_info for every valid name.")


def _masked(request: Request) -> dict:
    row = request.as_dict()
    match = HOUSE_NUMBER.match(request.address)
    if match:
        row["address"] = f"{int(match.group(1)) // 100 * 100} block of {match.group(2).strip()}"
    return row


def _matching(service: str | None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check(service, SERVICES, "service")
    _check(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if service in (None, r.service) and neighborhood in (None, r.neighborhood) and status in (None, r.status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_ROWS)] = 20,
) -> list[dict]:
    """List individual requests matching the filters, newest first.

    Use this to see examples or details. To answer "how many" or "which is
    most", use count_requests instead; for response times, median_close_hours.
    Service and neighborhood must be exact names from dataset_info.
    """
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [_masked(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: GroupBy | None = None,
) -> int | dict[str, int]:
    """Count requests matching the filters.

    Without group_by, returns one number. With group_by, returns a dict of
    group to count, largest first, at most 50 groups; use this for "which day,
    neighborhood, or service had the most". Use search_requests to see the
    requests themselves. Service and neighborhood must be exact names from
    dataset_info.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(service: str | None = None, group_by: MedianGroupBy | None = None) -> float | None | dict[str, float]:
    """Median hours from opened to closed, over closed requests, rounded to 0.1.

    Without group_by, returns one number, or null if nothing has closed. With
    group_by, returns a dict, fastest first, leaving out groups with fewer than
    10 closed requests. Use count_requests for volumes, not speed. Service must
    be an exact name from dataset_info.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: defaultdict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one request by its request_id, with every field.

    Use this to look up an ID seen in search_requests results. To find
    requests without an ID, use search_requests.
    """
    logger.info("get_request %s by %s", request_id, caller_email() or "local")
    if request_id not in BY_ID:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find real ids.")
    return _masked(BY_ID[request_id])


if __name__ == "__main__":
    mcp.run()
