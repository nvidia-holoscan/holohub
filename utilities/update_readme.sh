#!/usr/bin/bash

# Simple utility script to update GitHub badges in the HoloHub README.md file.
# Usage:
#
# ```bash
# ./utilities/update_readme.sh
# git add README.md && git commit -s -m "Update HoloHub Project Statistics"
# ```

python ./utilities/gather_metadata.py --output aggregate_metadata.json
app_count=$(cat aggregate_metadata.json | grep "\"source_folder\": \"applications\"" | wc -l)
sed -i -E "s/Applications-([0-9]+)/Applications-$app_count/" README.md
ops_count=$(cat aggregate_metadata.json | grep "\"source_folder\": \"operators\"" | wc -l)
sed -i -E "s/Operators-([0-9]+)/Operators-$ops_count/" README.md
tutorial_count=$(cat aggregate_metadata.json | grep "\"source_folder\": \"tutorials\"" | wc -l)
sed -i -E "s/Tutorials-([0-9]+)/Tutorials-$tutorial_count/" README.md
