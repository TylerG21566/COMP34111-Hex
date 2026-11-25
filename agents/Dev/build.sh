#!/bin/bash
# Builds the C++ agent to the path DevAgent.py launches: cpp_agent, next to
# this script. Paths are relative to this script, so it still works when this
# folder is renamed to GroupXYZ. For marked games, run it in the hex-build image.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sources=()
while IFS= read -r -d '' file; do
    sources+=("$file")
done < <(find "$here/TreeSearch/bot/src" "$here/TreeSearch/modes/VersusPlay" -name '*.cpp' -print0)

if [ "${#sources[@]}" -eq 0 ]; then
    echo "build.sh: no .cpp files in TreeSearch/bot/src or TreeSearch/modes/VersusPlay" >&2
    exit 1
fi

flags=(-std=c++20 -O3 -DNDEBUG -Wall -Wextra -pthread -I "$here/TreeSearch/bot/include")
# The marking machine is x86-64: never use -march=native (docs/GAME.md)
if [ "$(uname -m)" = "x86_64" ]; then
    flags+=(-march=x86-64-v3)
fi

"${CXX:-g++}" "${flags[@]}" -o "$here/cpp_agent" "${sources[@]}"
echo "built $here/cpp_agent"
