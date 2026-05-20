LOAD_CLASS_NAMES = {
    0: "NORMAL",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL",
}


def score_to_class(score: float) -> int:
    if score < 0.40:
        return 0
    if score < 0.60:
        return 1
    if score < 0.80:
        return 2
    return 3


def class_id_to_name(class_id: int) -> str:
    return LOAD_CLASS_NAMES.get(int(class_id), "UNKNOWN")


def _normalize_class_name(load_class: int | str) -> str:
    if isinstance(load_class, str):
        return load_class
    return class_id_to_name(int(load_class))


def recommended_status(current_load_class: int | str, future_load_class: int | str) -> str:
    current_name = _normalize_class_name(current_load_class)
    future_name = _normalize_class_name(future_load_class)

    if current_name == "CRITICAL":
        return "IMMEDIATE_ACTION_REQUIRED"
    if future_name == "CRITICAL":
        return "CRITICAL_RISK_PREDICTED"
    if future_name == "HIGH":
        return "PREVENTIVE_ACTION_REQUIRED"
    if current_name == "HIGH":
        return "OBSERVE_HIGH_LOAD"
    return "STABLE"
