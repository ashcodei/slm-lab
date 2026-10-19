"""Conservative internal problem grouping; not a contamination certificate."""
import re


class ProblemGroups:
    def __init__(self, problems):
        self.parent = {problem: problem for problem in problems}
        if not self.parent or any(not re.fullmatch(r'p[0-9]{5}', p) for p in self.parent):
            raise ValueError('Original problem identifiers required')

    def root(self, problem):
        if problem not in self.parent:
            raise ValueError('Unknown original problem identifier')
        current = problem
        while self.parent[current] != current:
            current = self.parent[current]
        while self.parent[problem] != problem:
            following = self.parent[problem]
            self.parent[problem] = current
            problem = following
        return current

    def merge(self, problems):
        roots = sorted({self.root(p) for p in problems})
        for root in roots[1:]:
            self.parent[root] = roots[0]
        return max(0, len(roots) - 1)

    def mapping(self):
        return {problem: self.root(problem) for problem in sorted(self.parent)}


def identical_groups(text):
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        members = [part.strip() for part in line.split(',')]
        if len(set(members)) < 2 or any(not re.fullmatch(r'p[0-9]{5}', p) for p in members):
            raise ValueError('Original identical-problem cluster format differs')
        result.append(tuple(members))
    return result


def canonical_member(value):
    # Publisher analyses retain the older CodeNet directory name.
    match = re.search(r'(?:^|/)(?:CodeNet|Project_CodeNet)/data/(p[0-9]{5})/(C\+\+|C|Java|Python)/(s[0-9]{9}\.[A-Za-z0-9_+#-]+)$', value.strip())
    if not match:
        raise ValueError('Original publisher source-member path format differs')
    return 'Project_CodeNet/data/' + '/'.join(match.groups())


def selected_clone_groups(lines, selected, language):
    """Yield selected members only, retaining original cluster boundaries."""
    current = set()
    active = False
    for line in lines:
        text = line.strip()
        if not text:
            if current:
                yield tuple(sorted(current))
            current = set()
            active = False
            continue
        name, colon, scores = text.partition(':')
        if not colon:
            raise ValueError('Original publisher cluster delimiter missing')
        member = canonical_member(name)
        if member.split('/')[3] != language:
            raise ValueError('Original publisher cluster language differs')
        if not scores.strip():
            if current:
                yield tuple(sorted(current))
            current = set()
            active = True
        elif not active:
            raise ValueError('Original publisher cluster lacks a representative')
        if member in selected:
            current.add(member)
    if current:
        yield tuple(sorted(current))
