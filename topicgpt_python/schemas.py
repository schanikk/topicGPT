from pydantic import BaseModel
from typing import List

class ExplainableTopic(BaseModel):
    """
    Model for structured output of topics
    """
    level: int
    name: str
    description: str

class ExplainableTopics(BaseModel):
    """
    Model for structured output of topics
    """
    topics: list[ExplainableTopic]

class TopicAssignment(BaseModel):
    """
    Model for structured output of topic assignments
    """
    level: int
    name: str
    description: str
    supporting_quote: str