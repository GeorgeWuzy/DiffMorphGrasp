"""Robot-hand names and shared validation helpers."""

SUPPORTED_HANDS = ("shadowhand", "allegro", "barrett", "leaphand", "inspire")


def validate_hand_name(hand_name: str) -> str:
    if hand_name not in SUPPORTED_HANDS:
        names = ", ".join(SUPPORTED_HANDS)
        raise ValueError(f"Unsupported hand '{hand_name}'. Choose one of: {names}.")
    return hand_name


def validate_hand_names(hand_names):
    return [validate_hand_name(name) for name in hand_names]
