#!/bin/bash

PROJECT_DIR="particle_physics_ml"
REPO_URL="https://github.com/julia1234456/particle_physics_ml"
COMMIT_DATE="2024-04-04T20:12:03"

cd "$PROJECT_DIR"

git init
git remote add origin "$REPO_URL"
# --- First commit (e.g. project start in 2024) ---
git add .
GIT_AUTHOR_DATE="$COMMIT_DATE" \
GIT_COMMITTER_DATE="$COMMIT_DATE" \
git commit -m "refactoring"


# --- Add more files over "time" ---
#git add src/
#GIT_COMMITTER_DATE="2024-02-20T14:00:00" \
#git commit -m "Add core features"

# --- A 2025 commit ---
#git add .
#GIT_AUTHOR_DATE="2025-06-01T11:00:00" \
#GIT_COMMITTER_DATE="2025-06-01T11:00:00" \
#git commit -m "Refactor and improvements"

# Push everything
git push -u origin main