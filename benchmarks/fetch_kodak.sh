#!/bin/sh
# Fetch the 24 Kodak images the comparison runs on, into datasets/kodak/.
set -e
dir="$(dirname "$0")/../datasets/kodak"
mkdir -p "$dir"
i=1
while [ "$i" -le 24 ]; do
    n=$(printf '%02d' "$i")
    if [ ! -f "$dir/kodim$n.png" ]; then
        curl -fsS -o "$dir/kodim$n.png" "https://r0k.us/graphics/kodak/kodak/kodim$n.png"
    fi
    i=$((i + 1))
done
echo "kodak images in $dir"
