#!/bin/bash
git update-index --skip-worktree 0_config_files/*
echo "Local config files are now ignored for updates. You can modify them without affecting git status."

