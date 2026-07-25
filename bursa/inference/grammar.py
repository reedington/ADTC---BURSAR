def _alt(values) -> str:
    return " | ".join(f'"\\"{v}\\""' for v in values)


def build_grammar(txn_id: str, candidate_ids: list[str], allowed_codes: list[str]) -> str:
    """GBNF restricting student_id + reason_codes to the supplied (post-ladder) values.
    transaction_id is fixed to the known literal. interpretation is a permissive object."""
    student_alt = _alt(candidate_ids) if candidate_ids else '"\\"\\""'
    code_alt = _alt(allowed_codes) if allowed_codes else '"\\"\\""'
    return f'''root ::= "{{" ws
  "\\"transaction_id\\":" ws "\\"{txn_id}\\"" ws "," ws
  "\\"interpretation\\":" ws object ws "," ws
  "\\"candidate_allocations\\":" ws "[" ws (alloc (ws "," ws alloc)*)? ws "]" ws "," ws
  "\\"recommended_action\\":" ws action ws "," ws
  "\\"explanation\\":" ws string ws "," ws
  "\\"ambiguities\\":" ws "[" ws (string (ws "," ws string)*)? ws "]" ws
  "}}" ws
alloc ::= "{{" ws "\\"student_id\\":" ws student_id ws "," ws "\\"amount_minor\\":" ws int ws "," ws "\\"reason_codes\\":" ws "[" ws (reason (ws "," ws reason)*)? ws "]" ws "}}"
student_id ::= {student_alt}
reason ::= {code_alt}
action ::= "\\"auto\\"" | "\\"review\\"" | "\\"unmatched\\""
int ::= [0-9]+
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
object ::= "{{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}}"
value ::= string | int | object | "[" ws (value (ws "," ws value)*)? ws "]" | "true" | "false" | "null"
ws ::= [ \\t\\n]*
'''
