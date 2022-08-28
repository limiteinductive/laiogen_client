import gc
import os
import threading
import time
import traceback
import urllib.request
from time import sleep

import GPUtil
import PIL.Image as Image
import torch
from returns.curry import partial
from returns.pipeline import flow, is_successful
from returns.pointfree import bind
from returns.result import Result, safe
from sdc.pipeline import initialize_plasma, run_stable_diffusion, StableDiffusionPlasma

from laiogen_client.database import (
    accept_next_job,
    connect_backend_database,
    get_job,
    update_error,
    update_job,
)
from laiogen_client.register import serve_vm_info

# from typing import TypeVar

# T = TypeVar('T')


def download_image(uri: str) -> Image.Image:
    urllib.request.urlretrieve(uri, "image.jpg")
    image = Image.open("image.jpg")

    return image


@safe
def run_job(plasma: StableDiffusionPlasma, job: JobArango) -> JobArango:

    if job.jobRequest.modelName == "glid-3-xl":
        job = run_ldm(plasma, job)
    elif job.jobRequest.modelName == "ldm":
        job = run_ldm(plasma, job)
    else:
        raise ValueError(f"Model name {job.Request.modelName} is unknow")

    return job


def run_ldm(plasma: StableDiffusionPlasma, job: JobArango):
    from googletrans import Translator

    translator = Translator()

    diffusion = job.jobRequest.inferenceConfig.diffusion
    text = translator.translate(diffusion.prompts[0].prompt).text
    print("---> Translated prompt: ", text)
    batch_size = min(job.jobRequest.inferenceConfig.batchSize, 1)
    guidance_scale = diffusion.guidanceScale
    width = diffusion.imageFormat.width
    height = diffusion.imageFormat.height
    steps = diffusion.steps
    skip_steps = diffusion.skipSteps / 100
    init_image = (
        download_image(diffusion.initImage.uri) if diffusion.initImage else None
    )

    outputs = run_stable_diffusion(
        plasma=plasma,
        prompt=text,
        steps=steps,
        skip_steps=skip_steps,
        init_image=init_image,
        width=width,
        height=height,
        batch_size=batch_size,
        checkpoint_path="./models/stable-diffusion.pt",
        guidance_scale=guidance_scale,
        device="cuda",
    )

    job.images = []
    for img in outputs["sample"]:
        image_uri, blur_hash = upload_image_to_storage(img)
        job.images.append({"uri": image_uri, "blurHash": blur_hash})

    job.completionDate = time.time()
    job.status = "completed"

    return job


def unwrap_error(self, db, job: JobArango) -> None:
    job = get_job(db, job._key)
    print("unwrap", job._key)
    if is_successful(self):
        print("---> Job completed.")
        print("success", job._key)
    else:
        print("error", job._key)
        try:
            print("try")
            raise self.failure()
        except:
            print("except")
            traceback_text = traceback.format_exc()

        print("after")
        job = update_error(db, job, str(self.failure()), traceback_text)
        print(f"---> Error in the job: {self.failure()}")
        print(traceback_text)

    del self
    torch.cuda.empty_cache()
    gc.collect()


Result.unwrap_error = unwrap_error


def run_laiogen(db, plasma, job) -> None:
    flow(
        job,
        partial(run_job, plasma),
    ).unwrap_error(db, job)


def main(verbose=True):
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    db = connect_backend_database()

    print("---> Initializing plasma...")
    plasma = initialize_plasma("./models/stable-diffusion.pt")

    print("---> Ready to work")

    while True:
        if GPUtil.getAvailable(order="first", limit=1, maxLoad=0.5, maxMemory=1.0):
            result = accept_next_job(db)

            if is_successful(result):
                job = result.unwrap()
                if job:
                    print("---> Found a job to proccess.")

                    th = threading.Thread(
                        run_laiogen,
                        args=(db, plasma, job),
                    )
                    th.run()
                else:
                    print("---> No job available, waiting 2 sec... ")

                sleep(2)
            else:
                raise result.failure()

        else:
            print("---> Currently, no GPU available, waiting 5 sec...")
            sleep(5)


if __name__ == "__main__":
    import multiprocessing as mp
    from loguru import logger

    user_id = "me"
    user_pwd = "test"

    mp.set_start_method("spawn")

    svi = mp.Process(
        target=serve_vm_info,
        kwargs={
            "user_id": user_id,
            "user_pwd": user_pwd,
            "refresh_rate": 1,
            "verbose": False,
        },
    )
    mni = mp.Process(target=main, kwargs={"verbose": True})
    svi.start()
    mni.start()
    svi.join()
    mni.join()
