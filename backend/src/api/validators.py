def get_simulate_request_type(data):
    """
    Auto-detect request type and inject request_type field.
    This function both detects AND modifies the incoming data.
    """

    # Handle raw dict (most common case)
    if isinstance(data, dict):
        if "prediction_id" in data:
            data["request_type"] = "prediction"
            return "prediction"
        else:
            data["request_type"] = "normal"
            return "normal"

    return "normal"


def to_dict(obj):
    if hasattr(obj, "model_dump"):  # Pydantic - clean,
        data = obj.model_dump()
    elif hasattr(obj, "__dict__"):  # Fallback for other
        data = obj.__dict__.copy()

        data.pop("_sa_instance_state", None)
    else:
        data = dict(obj)

    if "pid" not in data and "id" in data:
        data["pid"] = data["id"]

    return data
