# times.py — back-compat shim.
#
# The time utilities moved into the `dabidabi` package
# (packages/db/src/dabidabi/times.py). Aliased via sys.modules so existing
# `from src.infra.times import utc_now, db_now, ...` keeps working unchanged.
# Packages should import `dabidabi.times` directly.
import sys

import dabidabi.times

sys.modules[__name__] = dabidabi.times
