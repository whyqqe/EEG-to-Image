#!/usr/bin/env bash
# Time-budget helpers for the pipeline, factored out so they can be unit-tested
# on the login node (where no GPU is available and the full pipeline cannot run).
#
# Deliberately *functions writing to stdout* rather than globals: with `set -u`,
# a mistyped `${remaining}` silently becomes an "unbound variable" crash deep
# inside a running job rather than a visible error, so the call convention is
# kept to one obvious form -- $(remaining) -- and exercised by a test.

# Usage: guard::init <budget_seconds> <reserve_seconds> <start_epoch>
guard::init() {
  GUARD_START_EPOCH="$3"
  GUARD_BUDGET_S="$1"
  GUARD_RESERVE_S="$2"
}

guard::now() { date +%s; }

guard::elapsed_s() { echo $(( $(guard::now) - GUARD_START_EPOCH )); }

guard::remaining() {
  local r=$(( GUARD_BUDGET_S - ( $(guard::now) - GUARD_START_EPOCH ) ))
  echo "${r}"
}

# True when at least $1 seconds remain after reserving the summary budget.
guard::have_time_for() {
  local need="$1"
  local rem
  rem=$(guard::remaining)
  (( rem - GUARD_RESERVE_S > need ))
}

guard::elapsed_h() {
  awk "BEGIN{printf \"%.2f\", ($(guard::now)-${GUARD_START_EPOCH})/3600}"
}
