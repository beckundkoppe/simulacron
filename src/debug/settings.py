# debug.py
# Global debug debuguration

DEBUG_PERCEPTION_ENABLED = True

# Each entry is a tuple: (readable_id, person_name, datum_key)
# Use "*" as wildcard for a field you don't want to filter by.
DEBUG_PERCEPTION_FILTER: set[tuple[str, str, str]] = set()

DEBUG_PERCEPTION_FILTER.add(("*", "*", "material"))

VERBOSE_BACKEND = False
