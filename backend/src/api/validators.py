def get_simulate_request_type(data):
    """
    Auto-detect request type based on prediction_id.
    This function detects the incoming data.

    Returns 'prediction' if the data contains prediction_id, otherwise 'normal'.
    """

    # Handle raw dict (most common case)
    if isinstance(data, dict):
        if "prediction_id" in data:
            return "prediction"

    return "normal"


def to_dict(obj):
    if hasattr(obj, "model_dump"):  # Pydantic - clean,
        data = obj.model_dump()
    elif hasattr(obj, "__dict__"):  # Fallback for other
        data = obj.__dict__.copy()

        data.pop("_sa_instance_state", None)
    else:
        try:
            data = dict(obj)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot convert {type(obj).__name__} to dict") from e

    if "pid" not in data and "id" in data:
        data["pid"] = data["id"]

    return data
