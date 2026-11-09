#!/usr/bin/env bash
# Commit one round's IceLab-EDM file in the hub fork and open the pull request to RespiCast.
#
#   weekly/hub_pr.sh 2026-11-11            (DRY_RUN=1: only the checks)
#
# In ~/git/RespiCast-SyndromicIndicators on main (tracking origin/main, the fork albertasdvirnas/...):
# checks model-output/IceLab-EDM/<origin>-IceLab-EDM.csv (rows, no empty, negative or non-numeric
# values), then git fetch upstream, git merge upstream/main, commits the file as
# "icelab-edm new predictions <origin>" with the git identity configured there, git push origin main
# and gh pr create from albertasdvirnas:main (the logged-in gh account). It stops without changing
# anything if the fork has other changes, the file is already upstream or a PR from
# albertasdvirnas:main is still open, and stops before pushing if the PR's diff would be more than
# that one file. Prints the PR URL last. Called by respicast_weekly.sh after the copy-back.
set -euo pipefail
origin=${1:?usage: hub_pr.sh YYYY-MM-DD}
cd "$HOME/git/RespiCast-SyndromicIndicators"
UPSTREAM=european-modelling-hubs/RespiCast-SyndromicIndicators
FORK_OWNER=albertasdvirnas
file=model-output/IceLab-EDM/$origin-IceLab-EDM.csv
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}
export GH_PROMPT_DISABLED=1
die() { echo "hub_pr: $*" >&2; exit 1; }

[ -s "$file" ] || die "no $file"
awk -F, 'NR > 1 { n++; if ($8 !~ /^[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$/) bad++ }
         END { exit !(n > 0 && !bad) }' "$file" || die "$file has no rows or an empty, negative or non-numeric value"
[ "$(git branch --show-current)" = main ] || die "the fork is not on main"
other=$(git status --porcelain | grep -vxF "?? $file" || true)
[ -z "$other" ] || die "the fork has other changes: $other"
git ls-files --error-unmatch "$file" >/dev/null 2>&1 && die "$file is already committed"
git fetch --quiet upstream
git cat-file -e "upstream/main:$file" 2>/dev/null && die "$file is already on upstream/main"
open=$(gh pr list -R "$UPSTREAM" --state open --head main --json url,headRepositoryOwner \
       --jq ".[] | select(.headRepositoryOwner.login == \"$FORK_OWNER\") | .url")
[ -z "$open" ] || die "a PR from $FORK_OWNER:main is still open ($open); not adding to it"
counts=$(awk -F, 'NR > 1 { print $2 "," $5 }' "$file" | sort -u | cut -d, -f1 | uniq -c |
         awk '{ n = $1; $1 = ""; printf "%s%s for %d locations", (NR > 1 ? ", " : ""), substr($0, 2), n }')
if [ "${DRY_RUN:-0}" = 1 ]; then echo "dry run: checks passed ($counts); would merge, commit, push and open the PR"; exit 0; fi

git merge --no-edit upstream/main || { git merge --abort; die "merge of upstream/main failed"; }
git add "$file"
git commit --quiet -m "icelab-edm new predictions $origin"
diff=$(git diff --name-only upstream/main HEAD)
[ "$diff" = "$file" ] || die "PR diff would not be only $file: $diff (committed locally, not pushed)"

git push --quiet origin main
gh pr create -R "$UPSTREAM" --base main --head "$FORK_OWNER:main" \
    --title "icelab-edm new predictions $origin" \
    --body "IceLab-EDM forecasts for origin date $origin: $counts."
