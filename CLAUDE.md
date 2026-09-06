@AGENTS.md

## Claude Code
- Droplet commands: Claude has no SSH key for `146.190.76.14`. Ask the user to run them with the `!` prefix in this session so the output lands in the conversation.
- Deploys (`deploy/deploy.sh` → rsync + `systemctl restart`) restart live systemd trading services; they are soft-denied in auto mode and must be run by the user.
