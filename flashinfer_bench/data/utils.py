from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
"""Type alias for non-empty strings with minimum length of 1."""


class BaseModelWithDocstrings(BaseModel):
    """Base model with the attribute docstrings being extracted to the model JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)
