#!/usr/bin/env bash
# Sync local main with upstream HKUDS/nanobot, preserving local commits.
#
# Strategy: reset to upstream/main, then cherry-pick local-only commits.
# This preserves upstream's exact SHAs (including merge commits) so GitHub
# shows "N ahead, 0 behind" rather than duplicate diverged histories.
#
# Usage:
#   ./sync-upstream.sh          # fetch + sync only
#   ./sync-upstream.sh --push   # fetch + sync + push to origin

set -euo pipefail

UPSTREAM_URL="https://github.com/HKUDS/nanobot.git"
UPSTREAM_NAME="upstream"
BRANCH="main"

# --- helpers ---------------------------------------------------------------
info()  { printf '\033[1;34m→ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
err()   { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

# --- pre-checks -----------------------------------------------------------
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    err "Not inside a git repository"; exit 1
fi

if [ -n "$(git diff --stat)" ] || [ -n "$(git diff --cached --stat)" ]; then
    err "Working tree has uncommitted changes. Commit or stash first."; exit 1
fi

current=$(git branch --show-current)
if [ "$current" != "$BRANCH" ]; then
    info "Switching to $BRANCH"
    git checkout "$BRANCH"
fi

# --- ensure upstream remote ------------------------------------------------
if ! git remote get-url "$UPSTREAM_NAME" &>/dev/null; then
    info "Adding upstream remote: $UPSTREAM_URL"
    git remote add "$UPSTREAM_NAME" "$UPSTREAM_URL"
fi
ok "Upstream remote: $(git remote get-url $UPSTREAM_NAME)"

# --- fetch -----------------------------------------------------------------
info "Fetching $UPSTREAM_NAME/$BRANCH"
git fetch "$UPSTREAM_NAME" "$BRANCH"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "$UPSTREAM_NAME/$BRANCH")
BASE=$(git merge-base HEAD "$UPSTREAM_NAME/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    ok "Already up to date with upstream."; exit 0
fi

if [ "$LOCAL" = "$BASE" ]; then
    info "Fast-forwarding to upstream (no local-only commits)"
    git merge --ff-only "$UPSTREAM_NAME/$BRANCH"
    ok "Fast-forwarded to upstream"
elif [ "$REMOTE" = "$BASE" ]; then
    ok "Already ahead of upstream — nothing to sync."
    exit 0
else
    # Collect local-only commit SHAs (oldest first) for cherry-pick
    LOCAL_COMMITS=$(git rev-list --reverse "$UPSTREAM_NAME/$BRANCH"..HEAD)
    LOCAL_COUNT=$(echo "$LOCAL_COMMITS" | wc -l | tr -d ' ')
    UPSTREAM_NEW=$(git rev-list HEAD.."$UPSTREAM_NAME/$BRANCH" --count)
    info "Replaying $LOCAL_COUNT local commit(s) on top of $UPSTREAM_NEW new upstream commit(s)"

    # Reset to upstream, then cherry-pick local commits
    git reset --hard "$UPSTREAM_NAME/$BRANCH"

    if ! echo "$LOCAL_COMMITS" | xargs git cherry-pick; then
        err "Cherry-pick conflict! Resolve manually, then:"
        err "  git cherry-pick --continue"
        err "  git push --force-with-lease origin $BRANCH"
        err ""
        err "Or abort with: git cherry-pick --abort"
        exit 1
    fi
    ok "Sync successful — $LOCAL_COUNT local commit(s) replayed"
fi

# --- push ------------------------------------------------------------------
if [[ "${1:-}" == "--push" ]]; then
    info "Pushing to origin/$BRANCH (force-with-lease)"
    git push --force-with-lease origin "$BRANCH"
    ok "Pushed to origin/$BRANCH"
else
    info "Run with --push to push to origin"
fi

ok "Sync complete"
