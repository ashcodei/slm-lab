"""Trusted pytest hooks; original publisher tests/solutions are never rewritten."""
import json
import faulthandler
from pathlib import Path
import sys
faulthandler.enable()

# -I -S keeps ambient packages/plugins out. Only this pinned package directory
# and the explicitly supplied complete source tree are added afterwards.
settings=json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0,settings['packages'])
sys.path.insert(0,settings['source'])
import pytest
from _pytest.subtests import SubtestReport

PREFIX='SLM_PYTHON_PUBLISHER_INVENTORY_V2\t'


class Inventory:
    def __init__(self):
        self.registered=[];self.started=[];self.events=[];self.collection_errors=[];self.subtests={}

    def pytest_collection_finish(self,session):
        self.registered=[item.nodeid for item in session.items]

    def pytest_collectreport(self,report):
        if report.failed:self.collection_errors.append(dict(nodeid=report.nodeid,diagnostic=str(report.longrepr)))

    def pytest_runtest_logstart(self,nodeid,location):
        self.started.append(nodeid)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self,item,call):
        outcome=yield
        report=outcome.get_result()
        exception=call.excinfo.type if call.excinfo else None
        report.own_exception_type=(exception.__module__+'.'+exception.__qualname__) if exception else None
        report.own_assertion_failure=bool(exception and issubclass(exception,AssertionError))

    def pytest_runtest_logreport(self,report):
        subtest=isinstance(report,SubtestReport)
        index=None;context=None
        if subtest:
            index=self.subtests.get(report.nodeid,0)+1;self.subtests[report.nodeid]=index
            context=report.context._to_json()
        self.events.append(dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,
            kind='subtest' if subtest else 'parent',subtest_index=index,subtest_context=context,
            expected_failure=getattr(report,'wasxfail',None),exception_type=report.own_exception_type,
            assertion_failure=report.own_assertion_failure))


plugin=Inventory()
code=pytest.main(['-p','no:cacheprovider','-p','no:faulthandler','-p','no:logging','-p','no:capture',
    '--rootdir='+settings['source'],'--confcutdir='+settings['source'],'--import-mode=importlib',
    '--tb=short','--disable-warnings','--basetemp='+settings['temporary']]+settings['tests'],plugins=[plugin])
record=dict(format='own-complete-python-publisher-events-v2',framework='pytest',framework_version=pytest.__version__,
    registered=plugin.registered,started=plugin.started,events=plugin.events,
    collection_errors=plugin.collection_errors,session_exitstatus=int(code))
print(PREFIX+json.dumps(record,separators=(',',':')),flush=True)
