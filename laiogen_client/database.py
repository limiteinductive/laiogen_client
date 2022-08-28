import time
from typing import List
import GPUtil
import nanoid
from sdc.pipeline import StableDiffusionOutput

from laiogen_client.typing import Credentials, Job


def check_credentials(user_id: str, user_pwd: str) -> Credentials:
    time.sleep(2)
    user_name = "limiteinductive"

    return Credentials(user_id=user_id, user_pwd=user_pwd, user_name=user_name)


def get_vm_gpus_info() -> List[dict]:
    gpus = GPUtil.getGPUs()

    return list(map(lambda x: x.__dict__, gpus))


def register_gpus(credentials: Credentials) -> List[dict]:
    time.sleep(2)
    gpus = get_vm_gpus_info()

    return gpus


def accept_next_job(credentials: Credentials) -> Job:

    time.sleep(1)
    job_id = nanoid.generate()

    return Job(id=job_id, prompt="cute cat", batch_size=1, guidance_scale=7.0, width=512, height=512, steps=50, skip_steps=0.)

def upload_images(credentials: Credentials, job: Job, outputs: StableDiffusionOutput) -> Job:
    time.sleep(1)

    return job



def update_job(credentials: Credentials, job: Job) -> Job:
    time.sleep(1)

    return job
