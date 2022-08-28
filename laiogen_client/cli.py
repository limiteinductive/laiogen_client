import subprocess
import typer
from sdc.pipeline import run_stable_diffusion, initialize_plasma, StableDiffusionPlasma
from pathlib import Path
from loguru import logger
from googletrans import Translator
import multiprocessing as mp
import GPUtil
import threading
import ray
import time
from pydantic import BaseModel
from laiogen_client.database import (
    check_credentials,
    register_gpus,
    accept_next_job,
    upload_images,
    update_job,
)

from laiogen_client.typing import Job, Credentials

import warnings

warnings.filterwarnings("ignore")


app = typer.Typer()


@ray.remote(num_gpus=1)
def run_ldm(credentials: Credentials, plasma: StableDiffusionPlasma, job: Job, device="cuda"):
    try:
        translator = Translator()
        job.translated_prompt = translator.translate(job.prompt).text
    except:
        pass

    output = run_stable_diffusion(
        plasma=plasma,
        prompt=job.translated_prompt if job.translated_prompt else job.prompt,
        steps=job.steps,
        skip_steps=job.skip_steps,
        init_image=job.init_image,
        width=job.width,
        height=job.height,
        batch_size=job.batch_size,
        guidance_scale=job.guidance_scale,
        device=device,
        verbose=False,
    )

    job.completion_time = time.time()
    job.status = "completed"

    job = upload_images(credentials, job, output)
    job = update_job(credentials, job)


def check_weights_path() -> Path:
    weights_path = Path.home() / ".cache/stable-diffusion/"
    weights_path.mkdir(exist_ok=True)
    if not (weights_path / "stable-diffusion.pt").is_file():
        logger.info(f"Downloading stable-diffusion weights to {weights_path}")
        subprocess.run(
            [
                "wget",
                "-P",
                str(weights_path),
                "https://storage.googleapis.com/laion_limiteinducive/stable-diffusion.pt",
            ]
        )

    return weights_path / "stable-diffusion.pt"


@app.command()
def main(user_id: str = typer.Option(..., prompt=True), user_pwd: str = typer.Option(..., prompt=True, hide_input=True )):
    ray.init()

    logger.add("laiogen_client.log", rotation="20 MB")

    logger.info("Checking credentials.")
    credentials = check_credentials(user_id, user_pwd)

    logger.info(f"Connected as {credentials.user_name}")

    gpus_info = register_gpus(credentials)
    logger.info(f"Found {len(gpus_info)} GPUs.")

    weights_path = check_weights_path()

    logger.info("Loading models into the plasma.")
    plasma = initialize_plasma(weights_path)

    logger.info("Ready to work.")

    while True:
        if GPUtil.getAvailable(order="first", limit=100, maxLoad=0.5, maxMemory=1.0):
            job: Job = accept_next_job(credentials)

            if job:
                logger.info(f"Got job {job.id}. Running...")
                job.acceptance_time = time.time()
                job.status = "running"
                job = update_job(credentials, job)

                future = run_ldm.remote(credentials, plasma, job)

                th = threading.Thread(
                    target=lambda x: ray.get(future),
                )

            else:
                logger.info("No job available, waiting 2 sec...")

            time.sleep(2)
        else:
            logger.info("---> Currently, no GPU available, waiting 5 sec...")
            time.sleep(5)
