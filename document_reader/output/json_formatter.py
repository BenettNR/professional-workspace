import json
from decimal import Decimal
from datetime import date, datetime

from extraction.models import ProcessingResult


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def to_json(result: ProcessingResult, indent: int = 2) -> str:
    return json.dumps(result.model_dump(), cls=_Encoder, indent=indent)
