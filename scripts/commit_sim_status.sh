#!/bin/bash

#git add .
git pull -r
git commit -a --amend --no-edit
git push -f
