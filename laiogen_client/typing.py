from pydantic import BaseModel
from typing import Optional, Literal, List


class Credentials(BaseModel):
    user_id: str
    user_pwd: str
    user_name: str


class Job(BaseModel):
    status: Literal[
        "pending", "running", "completed", "failed", "cancelled"
    ] = "pending"
    id: Optional[str]
    user_id: Optional[str]
    gpu_id: Optional[str]
    submission_time: Optional[float]
    acceptance_time: Optional[float]
    completion_time: Optional[float]
    prompt: str
    translated_prompt: Optional[str]
    batch_size: int
    guidance_scale: float
    width: int
    height: int
    steps: int
    skip_steps: float
    images: List[str]
    init_image: Optional[str]
