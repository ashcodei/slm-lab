"""Wait through transient atomic checkpoint space while retaining the reserve."""
import math
import shutil
import time


def wait_for_reserve(root,*,minimum_bytes,maximum_seconds=1800,interval_seconds=5,
                     guard=lambda:None,free_bytes=None,now=None,pause=None):
    if (type(minimum_bytes) is not int or minimum_bytes<1
            or type(maximum_seconds) not in {int,float} or not math.isfinite(maximum_seconds) or not 0<maximum_seconds<=1800
            or type(interval_seconds) not in {int,float} or not math.isfinite(interval_seconds) or not 0<interval_seconds<=30
            or not callable(guard)):
        raise ValueError('An explicit positive reserve and bounded wait are required')
    free_bytes=free_bytes or (lambda:shutil.disk_usage(root).free)
    now=now or time.monotonic;pause=pause or time.sleep;started=now();polls=0
    while True:
        guard();available=free_bytes()
        if type(available) is not int or available<0:raise ValueError('An actual finite nonnegative free-byte observation is required')
        if available>=minimum_bytes:return dict(waited_seconds=now()-started,failed_space_observations=polls,final_free_bytes=available)
        remaining=maximum_seconds-(now()-started)
        if remaining<=0:raise ValueError('Original disk reserve remains unavailable after the bounded wait')
        polls+=1;pause(min(interval_seconds,remaining))
