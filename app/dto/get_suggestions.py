from typing import List

from pydantic import BaseModel

class Property(BaseModel):  
    ref: str
    code_departement: int
    description: str
    cluster: int
    coords: str

class GetSuggestionsRequest(BaseModel):
    properties_user_preferences: List[Property]
    properties_by_cluster: List[Property]
    nbr_similar_property: int


class GetSuggestionsResponse(BaseModel):
    properties_to_suggest: List[Property]