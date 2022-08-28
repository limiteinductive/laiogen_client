from typing import List
import asyncio
import time
import GPUtil
from returns.result import Result


def get_vm_gpus_info() -> List[dict]:
    gpus = GPUtil.getGPUs()

    return list(map(lambda x: x.__dict__, gpus))


def serve_vm_info(user_id: str, user_pwd: str, refresh_rate=1, verbose=False) -> None:
    async def main():
        while True:
            vm_info = get_vm_info()

            if is_successful(update_vm_info(db, vm_info)):
                if verbose:
                    print("-> VM info published")
            else:
                print("-> WARNING VM info was not published")

            await asyncio.sleep(refresh_rate)

    asyncio.run(main())
