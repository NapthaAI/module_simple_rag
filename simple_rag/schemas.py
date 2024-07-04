from pydantic import BaseModel
from typing import Optional

class InputSchema(BaseModel):
    question: str
    input_dir: Optional[str] = None
