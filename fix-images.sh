#!/bin/bash

for f in "$@"; do
    convert -background \#ffffff -extent 0x0 "$f" "${f%.png}-fixed.png"
done
