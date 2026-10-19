"""Matched complete-module C++ retention checks for policy handoffs."""


def compare_reports(before, after):
    for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                'inference_precision', 'inference_device_type', 'protocol', 'toolchain', 'examples',
                'reference_report_sha256']:
        if before[key] != after[key]: raise ValueError('Unmatched C++ exercise assessment: ' + key)

    def cases(report):
        rows = report['results']
        if not rows or len(rows) != report['examples'] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Incomplete or repeated C++ exercise cases')
        compilation = sum(row['candidate']['compilation_succeeded'] for row in rows) / len(rows)
        functional = sum(row['candidate']['published_tests_passed'] for row in rows) / len(rows)
        if compilation != report['compilation_success_rate'] or functional != report['published_suite_pass_rate']:
            raise ValueError('C++ exercise aggregates differ from their case evidence')
        if any(row['candidate']['published_tests_passed'] and not row['candidate']['compilation_succeeded'] for row in rows):
            raise ValueError('C++ test pass without successful compilation')
        return [(row['id'], row['group_id'], row['prompt_sha256'], row['context_tokens'], row['generation_budget'],
                 row['solution_files'], row['test_sha256'], row['reference_inventory']) for row in rows]

    if cases(before) != cases(after): raise ValueError('Unmatched C++ exercise cases or generation budgets')
    lost = [old['id'] for old, new in zip(before['results'], after['results'])
            if old['candidate']['published_tests_passed'] and not new['candidate']['published_tests_passed']]
    return {'passed': not lost and after['compilation_success_rate'] >= before['compilation_success_rate']
            and after['published_suite_pass_rate'] >= before['published_suite_pass_rate'],
            'previously_correct_modules_lost': lost,
            'before_compilation_success_rate': before['compilation_success_rate'],
            'after_compilation_success_rate': after['compilation_success_rate'],
            'before_published_suite_pass_rate': before['published_suite_pass_rate'],
            'after_published_suite_pass_rate': after['published_suite_pass_rate'],
            'base_checkpoint_sha256': before['checkpoint_sha256'],
            'candidate_checkpoint_sha256': after['checkpoint_sha256'], 'goal_achieved': False,
            'scope': 'Matched complete C++ library modules and all pinned publisher tests; no efficiency or broad capability claim.'}
