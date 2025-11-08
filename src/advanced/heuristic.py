import ast
import json
import re
from typing import Any, Dict, List, Union


def _split_args(arg_str: str) -> List[str]:
    """Split a comma-separated argument string, respecting quotes and parentheses."""
    parts, buf, in_s, in_d, depth = [], [], False, False, 0
    for ch in arg_str:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == '(' and not in_s and not in_d:
            depth += 1
        elif ch == ')' and not in_s and not in_d and depth > 0:
            depth -= 1
        if ch == ',' and not in_s and not in_d and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]

def _coerce(value_str: str) -> Any:
    """Coerce a literal string to Python value if possible; else return as string (stripped quotes if present)."""
    v = value_str.strip()
    try:
        # ast.literal_eval handles numbers, strings, booleans, None, tuples, lists, dicts
        return ast.literal_eval(v)
    except Exception:
        # Fallback: bare identifiers -> keep as string
        return v.strip('"').strip("'")

def parse_toolcalls_fallback(text: str) -> Dict[str, Any]:
    """
    Parse the first function-call snippet into a single toolcall dict:
    {'name': <func>, 'args': [{'name': <arg>, 'value': <val>}, ...]}
    Positional args are ignored.
    """
    pattern = re.compile(r'([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)', re.DOTALL)
    m = pattern.search(text)
    if not m:
        raise ValueError("No function call found.")

    func, argblob = m.group(1), m.group(2)
    args_list: List[Dict[str, Any]] = []

    if argblob:
        for item in _split_args(argblob):
            if '=' not in item:
                continue  # ignore positional args
            name, val = item.split('=', 1)
            args_list.append({'name': name.strip(), 'value': _coerce(val)})

    return {'name': func, 'args': args_list}


def _dict_params_to_args(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Convert {"x": 0.0, "y": 1.0} -> [{"name":"x","value":0.0}, {"name":"y","value":1.0}]
    return [{'name': k, 'value': v} for k, v in params.items()]

def _normalize_one(call: Dict[str, Any]) -> Dict[str, Any]:
    # Handle shapes like {"name":..., "parameters": {...}} or {"name":..., "arguments": "..."}
    name = call.get('name') or call.get('function', {}).get('name')
    if not name:
        raise ValueError("Missing tool/function name.")

    # parameters may be dict; arguments may be JSON string or dict
    params = call.get('parameters')
    if params is None:
        args_raw = call.get('arguments') or call.get('function', {}).get('arguments')
        if isinstance(args_raw, str):
            try:
                params = json.loads(args_raw)  # parse JSON string payloads
            except json.JSONDecodeError:
                raise ValueError("arguments is not valid JSON.")
        elif isinstance(args_raw, dict):
            params = args_raw
        elif args_raw is None:
            params = {}
        else:
            raise ValueError("Unsupported arguments type.")

    if not isinstance(params, dict):
        raise ValueError("parameters/arguments must be an object.")

    return {'name': name, 'args': _dict_params_to_args(params)}

def parse_toolcall_json(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse a JSON payload into a single toolcall dict:
    {'name': <str>, 'args': [{'name': <str>, 'value': <any>}, ...]}.

    Accepts:
      - {"tool_call": {...}}
      - {"tool_calls": [ ... ]}   -> uses the first element
      - {"name": "...", "parameters": {...}}
      - {"function": {"name": "...", "arguments": "..."}}
    """
    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise ValueError("Top-level payload must be an object.")

    if 'tool_call' in payload:
        return _normalize_one(payload['tool_call'])

    if 'tool_calls' in payload:
        calls = payload['tool_calls']
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls must be a non-empty list.")
        return _normalize_one(calls[0])

    if any(k in payload for k in ('name', 'function', 'parameters', 'arguments')):
        return _normalize_one(payload)

    raise ValueError("Unrecognized tool-call JSON structure.")
