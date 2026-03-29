from __future__ import annotations


def parse_case_indices(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple, set)):
        values = [int(index) for index in raw_value]
    else:
        values = []
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            values.append(int(item))
    return sorted(set(values)) if values else None
