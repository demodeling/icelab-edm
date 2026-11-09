#!/usr/bin/env bash
# Weekly RespiCast round, started by systemd user units:
#   respicast-weekly.path   - whenever ~/.ssh/cm changes (the HPC2N ControlMaster socket is opened)
#   respicast-weekly.timer  - hourly 07:17-22:17, Saturday to Wednesday, in case the socket was already open
# It does nothing (exits silently) unless it is Saturday-Wednesday, this round's <origin>-IceLab-EDM.csv
# is not yet in ~/git/Respi, the round's truth data is out and an HPC2N socket in ~/.ssh/cm is alive.
# The hub moves forecasting_weeks.csv to the new round on Friday evening, a few hours before the
# truth data, so the data counts as out once the newest truth_date of both latest-ARI_ and
# latest-ILI_incidence.csv has reached the round's horizon-0 week (true of every round since 2026-09).
# If the data is out but no socket is alive, it sends a reminder, once a day. Otherwise it:
#   1. copies the newest run_final_YY_MM_DD.ipynb on HPC2N to today's name and runs it with
#      `sbatch run_final.sbatch` (Skylake node, works from either login node), waits for the job;
#   2. copies the notebook and CSV back to ~/git/Respi and the CSV into the hub fork's
#      model-output/IceLab-EDM/;
#   3. weekly/hub_pr.sh <origin>: in the hub fork, git fetch upstream, git merge upstream/main,
#      commit the CSV, git push origin main, gh pr create (the user's git identity and gh login);
#      if that fails, the run goes on and the notification says to run it by hand;
#   4. runs headless Claude (weekly/claude_prompt.md) to check the file, update the
#      "RespiCast progress" doc and the memory notes;
#   5. sends a desktop notification.
# Log: ~/git/Respi/weekly/logs/<origin>.log.  Dry run: DRY_RUN=1 respicast_weekly.sh
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}"

RESPI="${RESPI:-$HOME/git/Respi}"   # override only for testing
W="$HOME/git/Respi/weekly"
HUBFORK="$HOME/git/RespiCast-SyndromicIndicators/model-output/IceLab-EDM"
RAW='https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/blob/main'
WEEKS_URL="$RAW/supporting-files/forecasting_weeks.csv?raw=true"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$W/logs"

say() { echo "[$(date '+%F %T')] $*"; }
notify() { notify-send -a RespiCast "$@" 2>/dev/null || true; }

# One run at a time.
exec 9>"$W/.lock"
flock -n 9 || exit 0

# Saturday (6) to Wednesday (3) only.
case "$(date +%u)" in 4|5) [ "$DRY_RUN" = 1 ] && say "not Sat-Wed, exit"; exit 0 ;; esac

# This round's origin date (dfv.index[2] of the hub's forecasting_weeks.csv, 4th line) and horizon-0 week.
weeks="$(curl -fsSL "$WEEKS_URL")" || exit 0      # offline: try again at the next trigger
origin="$(awk -F, 'NR==4{print $1}' <<<"$weeks")"
h0="$(awk -F, '$2==0 && $4=="True"{print $5; exit}' <<<"$weeks")"
date_re='^20[0-9]{2}-[0-9]{2}-[0-9]{2}$'
[[ "$origin" =~ $date_re && "$h0" =~ $date_re ]] || { notify -u critical "RespiCast weekly run" "Could not read the origin date from forecasting_weeks.csv"; exit 1; }
csv="$origin-IceLab-EDM.csv"
[ -f "$RESPI/$csv" ] && { [ "$DRY_RUN" = 1 ] && say "$csv already in $RESPI, exit"; exit 0; }

# The round's truth data: the newest truth_date of both targets must reach the horizon-0 week.
newest() { curl -fsSL "$RAW/target-data/latest-$1_incidence.csv?raw=true" | awk -F, 'NR>1 && $3>m{m=$3} END{print m}'; }
ari="$(newest ARI)" || exit 0
ili="$(newest ILI)" || exit 0
if [[ "$ari" < "$h0" || "$ili" < "$h0" ]]; then
    [ "$DRY_RUN" = 1 ] && say "truth data for $origin not out yet (ARI $ari, ILI $ili; needs $h0), exit"
    exit 0
fi

# A live HPC2N ControlMaster socket; a socket that was just created may need a moment.
sock="" host=""
for attempt in $(seq 1 10); do
    found_file=0
    for s in "$HOME"/.ssh/cm/albertas@kebnekaise*.hpc2n.umu.se:22; do
        [ -S "$s" ] || continue
        found_file=1
        h="${s##*/}"; h="${h%:22}"
        if ssh -o BatchMode=yes -o ControlPath="$s" -O check "$h" >/dev/null 2>&1; then
            sock="$s"; host="$h"; break 2
        fi
    done
    [ "$found_file" = 1 ] || break
    sleep 30
done
if [ -z "$sock" ]; then
    if [ "$DRY_RUN" = 1 ]; then say "truth data for $origin is out (ARI $ari, ILI $ili); no live HPC2N socket, exit"
    elif [ "$(cat "$W/logs/.reminded" 2>/dev/null)" != "$origin $(date +%F)" ]; then
        echo "$origin $(date +%F)" >"$W/logs/.reminded"
        notify "RespiCast round $origin: data is out" "Open the HPC2N socket to start the run."
    fi
    exit 0
fi
R() { ssh -o BatchMode=yes -o ControlPath="$sock" "$host" "$@"; }

log="$W/logs/$origin.log"
[ "$DRY_RUN" = 1 ] || exec >>"$log" 2>&1
say "round with origin $origin (truth data to ARI $ari, ILI $ili); socket $sock"
trap 'say "FAILED (line $LINENO)"; notify -u critical "RespiCast weekly run failed" "Origin $origin. Log: $log"' ERR

nb="run_final_$(date +%y_%m_%d).ipynb"
job=""
if R "test -f git/Respi/$csv"; then
    say "$csv already exists on HPC2N: not rerunning (run_analysis appends), copying it back"
    nb="$(R 'cd ~/git/Respi && ls run_final_??_??_??.ipynb | sort | tail -1')"
    if [ "$DRY_RUN" = 1 ]; then say "dry run: would copy back $nb and $csv, open the hub PR, then run claude -p"; exit 0; fi
else
    prev="$(R 'cd ~/git/Respi && ls run_final_??_??_??.ipynb | sort | tail -1')"
    say "copying $prev to $nb and submitting run_final.sbatch"
    if [ "$DRY_RUN" = 1 ]; then say "dry run: stop before sbatch"; exit 0; fi
    [ "$prev" = "$nb" ] || R "cd ~/git/Respi && cp $prev $nb"
    job="$(R "bash -lc 'cd ~/git/Respi && sbatch --parsable run_final.sbatch $nb'")"
    job="${job%%;*}"
    say "job $job submitted"
    state=""
    for i in $(seq 1 240); do               # up to 2 hours (queue + ~3 min run)
        sleep 30
        state="$(R "bash -lc 'sacct -j $job -X -n -o State%20'" 2>/dev/null | tr -d ' ' || true)"
        case "$state" in
            COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) break ;;
        esac
    done
    say "job $job state: ${state:-unknown}"
    if [ "$state" != COMPLETED ]; then
        # A partial CSV would block the next attempt and append on a rerun: move it aside.
        R "cd ~/git/Respi && [ ! -f $csv ] || mv $csv $csv.failed_$job" || true
        R "tail -20 ~/git/Respi/logs/run_final_$job.out" || true
        notify -u critical "RespiCast weekly run failed" "Job $job ended ${state:-unknown}. Log: $log"
        exit 1
    fi
fi

scp -q -o BatchMode=yes -o ControlPath="$sock" "$host:git/Respi/$nb" "$host:git/Respi/$csv" "$RESPI/"
say "copied $nb and $csv to $RESPI"

# The hub PR, as the user (weekly/hub_pr.sh); a failure does not stop the doc update.
pr="not opened (hub_pr.sh failed, see the script log)"
if [ -d "$HUBFORK" ]; then
    cp "$RESPI/$csv" "$HUBFORK/" && say "copied $csv to $HUBFORK"
    if pr_out="$("$W/hub_pr.sh" "$origin" 2>&1)"; then
        echo "$pr_out"; pr="$(tail -1 <<<"$pr_out")"; say "hub PR opened: $pr"
    else
        echo "$pr_out"; say "hub_pr.sh failed"
        notify -u critical "RespiCast round $origin: PR not opened" "See $log, then run weekly/hub_pr.sh $origin"
    fi
fi

# Check the file, update the progress doc and memory with headless Claude.
prompt="$(sed -e "s/{{ORIGIN}}/$origin/g" -e "s/{{NOTEBOOK}}/$nb/g" -e "s/{{JOB}}/${job:-none}/g" \
              -e "s|{{LOG}}|$log|g" -e "s/{{HOST}}/$host/g" -e "s|{{PR}}|$pr|g" "$W/claude_prompt.md")"
say "starting claude -p"
summary_file="$W/logs/$origin.summary.txt"
cd "$RESPI"
claude -p "$prompt" \
    --permission-mode auto \
    --allowedTools "Read" "Write" "Edit" \
        "Bash(python3:*)" "Bash(curl:*)" "Bash(gh pr list:*)" "Bash(gh pr view:*)" \
        "Bash(ls:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(cat:*)" \
        "mcp__claude_ai_Claude_Docs__guide" "mcp__claude_ai_Claude_Docs__read" \
        "mcp__claude_ai_Claude_Docs__update" "mcp__claude_ai_Claude_Docs__batch" \
        "mcp__claude_ai_Claude_Docs__query" \
    --disallowedTools "Bash(gh pr create:*)" "Bash(gh pr merge:*)" "Bash(git commit:*)" "Bash(git push:*)" \
        "Bash(sbatch:*)" "mcp__claude_ai_Claude_Docs__delete" \
    > "$summary_file" || say "claude -p exited with $?"
cat "$summary_file"
say "done"
notify "RespiCast round $origin done" "PR: $pr. $(tail -c 300 "$summary_file")"
