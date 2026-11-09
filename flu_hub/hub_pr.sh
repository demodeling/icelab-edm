#!/bin/bash
# Commit one round's forecast in hub/ and open the pull request to the hub (notes/weekly-pr.md).
#
#   models/hub_pr.sh 2026-11-08
#
# In hub/ on main (tracking fork/main): re-validates model-output/dvirnas-edm/<date>-dvirnas-edm.csv,
# merges upstream/main, commits the file as "dvirnas-edm new predictions <date>" with the git identity
# configured there, pushes to fork/main and opens a PR from albertasdvirnas:main with gh (the logged-in
# account). It stops without changing anything if the hub clone has other changes or if a PR from
# albertasdvirnas:main is already open, and stops before pushing if the PR's diff would be more than
# that one file. Prints the PR URL last.

set -euo pipefail
ref=${1:?usage: hub_pr.sh YYYY-MM-DD}
cd "$(dirname "$0")/../hub"
UPSTREAM=philipgerlee/swedish_forecasting_hub
FORK_OWNER=albertasdvirnas
file=model-output/dvirnas-edm/$ref-dvirnas-edm.csv
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}
export GH_PROMPT_DISABLED=1
die() { echo "hub_pr: $*" >&2; exit 1; }

[ -e "$file" ] || die "no $file"
[ "$(git branch --show-current)" = main ] || die "hub/ is not on main"
other=$(git status --porcelain | grep -vxF "?? $file" || true)
[ -z "$other" ] || die "hub/ has other changes: $other"
git ls-files --error-unmatch "$file" >/dev/null 2>&1 && die "$file is already committed"

open=$(gh pr list -R "$UPSTREAM" --state open --head main --json url,headRepositoryOwner \
       --jq ".[] | select(.headRepositoryOwner.login == \"$FORK_OWNER\") | .url")
[ -z "$open" ] || die "a PR from $FORK_OWNER:main is still open ($open); not adding to it"

~/git/env/respi/bin/validate-forecast-submission "$file"

git fetch --quiet upstream
git merge --no-edit upstream/main || { git merge --abort; die "merge of upstream/main failed"; }
git add "$file"
git commit --quiet -m "dvirnas-edm new predictions $ref"

diff=$(git diff --name-only upstream/main HEAD)
[ "$diff" = "$file" ] || die "PR diff would not be only $file: $diff (committed locally, not pushed)"

git push --quiet fork main
gh pr create -R "$UPSTREAM" --base main --head "$FORK_OWNER:main" \
    --title "dvirnas-edm new predictions $ref" \
    --body "Forecast for the round of $ref: SE, SE-M and SE-O, horizons 0-3, from target data through $(date -d "$ref -7 days" +%F). Passes the hub validator."
