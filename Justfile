set dotenv-load

default:
    @just --list

sync-upstream:
    git fetch upstream
    git checkout main
    git merge upstream/main
    git push origin main
    git checkout dev-setup
    git rebase main