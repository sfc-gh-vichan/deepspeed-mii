from typing import List
from pydantic import BaseModel

class ModelInferRequest(BaseModel):
    prompts: str
    max_new_tokens: int | None = 128
    top_p: float = 0.0
    top_k: int | None = None
    temperature: int | None = None
    do_sample: bool = False
