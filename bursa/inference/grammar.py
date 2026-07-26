def _alt(values) -> str:
    return " | ".join(f'"\\"{v}\\""' for v in values)


def build_grammar(txn_id: str, candidate_ids: list[str], allowed_codes: list[str]) -> str:
    """GBNF restricting student_id + reason_codes to the supplied (post-ladder) values.
    transaction_id is fixed to the known literal. Every collection is cardinality-bounded so
    a small model cannot spend the complete 512-token output budget repeating valid items."""
    student_alt = _alt(candidate_ids) if candidate_ids else '"\\"\\""'
    code_alt = _alt(allowed_codes) if allowed_codes else '"\\"\\""'
    alloc_list = (
        f'alloc (ws "," ws alloc){{0,{len(candidate_ids) - 1}}}'
        if candidate_ids
        else ""
    )
    # GBNF rules are newline-terminated. Keep the complete root production on one physical
    # line; newer llama.cpp parsers correctly reject the old visually-wrapped form.
    root = " ".join([
        'root ::= "{" ws',
        f'"\\"transaction_id\\":" ws "\\"{txn_id}\\"" ws "," ws',
        '"\\"interpretation\\":" ws interpretation ws "," ws',
        '"\\"candidate_allocations\\":" ws "[" ws alloc-list? ws "]" ws "," ws',
        '"\\"recommended_action\\":" ws action ws "," ws',
        '"\\"explanation\\":" ws explanation-string ws "," ws',
        '"\\"ambiguities\\":" ws "[" ws short-list? ws "]" ws',
        '"}" ws',
    ])
    return f'''{root}
alloc-list ::= {alloc_list}
alloc ::= "{{" ws "\\"student_id\\":" ws student-id ws "," ws "\\"amount_minor\\":" ws int ws "," ws "\\"reason_codes\\":" ws "[" ws reason-list? ws "]" ws "}}"
reason-list ::= reason (ws "," ws reason){{0,3}}
short-list ::= short-string (ws "," ws short-string){{0,2}}
interpretation ::= "{{" ws "\\"payer_name\\":" ws short-string ws "," ws "\\"student_mentions\\":" ws "[" ws short-list? ws "]" ws "," ws "\\"term\\":" ws short-string ws "," ws "\\"fee_types\\":" ws "[" ws short-list? ws "]" ws "," ws "\\"payment_intent\\":" ws short-string ws "}}"
student-id ::= {student_alt}
reason ::= {code_alt}
action ::= "\\"auto\\"" | "\\"review\\"" | "\\"unmatched\\""
int ::= [0-9]+
short-string ::= "\\"" ([^"\\\\] | "\\\\" .){{0,64}} "\\""
explanation-string ::= "\\"" ([^"\\\\] | "\\\\" .){{0,240}} "\\""
ws ::= [ \\t\\n]?
'''
