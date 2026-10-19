"""Qualify one inspected incomplete terminal source record without inventing an ID."""


class QualifiedPublisherLines:
    def __init__(self, lines, expected_terminal_fragment):
        self.lines = iter(lines)
        self.expected_terminal_fragment = expected_terminal_fragment
        self.incomplete_terminal_record = None

    def __iter__(self):
        for number, line in enumerate(self.lines, 1):
            if line == self.expected_terminal_fragment and not line.endswith('\n'):
                if next(self.lines, None) is not None:
                    raise ValueError('Inspected truncated fragment is not the terminal record')
                self.incomplete_terminal_record = {'line': number, 'fragment': line,
                                                  'source_coverage_complete': False}
                return
            yield line
