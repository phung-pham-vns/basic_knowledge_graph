import os
import asyncio
from collections.abc import Coroutine
from typing import Any


SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 1))


# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max_coroutines or SEMAPHORE_LIMIT)

    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))
