#!/bin/bash
# Cron job: the live round for the coming Sunday once the week's target data is on upstream/main,
# then the pull request to the hub.
#
# crontab -e:
#   37 9-21 * * 4-6 /home/albertas/git/national-flu-hub/models/weekly_round.sh
# (hourly 09:37-21:37, Thursday to Saturday. The hub polls Folkhälsodata at :17 past the hour from
# Thursday; the data for 2026W42-W44 was merged on Thursday afternoons.)
#
# Runs models/live_round.py for next Sunday (or the date given as $1): it writes the forecast to
# submissions/ and hub/model-output/ and checks it with the hub validator. Then models/hub_pr.sh
# commits it in hub/ and opens the PR (notes/weekly-pr.md). Logs to logs/live-round-<date>.log.
# Once the round's file exists, later runs do nothing; if the PR step failed, fix it and run
# models/hub_pr.sh <date> by hand.
# A desktop notification gives the PR link, or reports a failure, or no data by the last run on Friday and Saturday.

cd "$(dirname "$0")/.." || exit 1
ref=${1:-$(date -d 'next sunday' +%F)}
log=logs/live-round-$ref.log
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}
notify() { notify-send -a flu-hub "$1" "$2" 2>/dev/null; }

[ -e "submissions/dvirnas-edm/$ref-dvirnas-edm.csv" ] && exit 0
mkdir -p logs
exec 9>logs/.live-round.lock
flock -n 9 || exit 0

{ echo "== $(date -Is)"; ~/git/env/respi/bin/python models/live_round.py "$ref"; } >>"$log" 2>&1
status=$?
case $status in
    0)  { echo "== $(date -Is) hub_pr"; models/hub_pr.sh "$ref"; } >>"$log" 2>&1
        status=$?
        if [ $status -eq 0 ]; then
            notify "dvirnas-edm $ref: PR opened" "$(tail -1 "$log")"
        else
            notify "dvirnas-edm $ref: forecast written, PR not opened" "See $log, then run models/hub_pr.sh $ref"
        fi ;;
    3)  [ "$(date +%u)" -ge 5 ] && [ "$(date +%H)" -ge 21 ] &&
            notify "dvirnas-edm $ref: no data yet" "The week's target data is not on upstream/main. Log: $log" ;;
    *)  notify "dvirnas-edm $ref: failed (exit $status)" "See $log" ;;
esac
exit $status
