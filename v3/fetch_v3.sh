#!/usr/bin/env bash
# Fetches RocketSim V3 (MIT; https://github.com/ZealanL/RocketSim, branch v3-rust) at the pinned commit into
# vendor/rocketsim-<rev>/, verifies the tree hash, applies v3/rocketsim-dump.patch (the state dump and phase-split
# stepping the tests use) and writes a marker. Works in Git Bash on Windows and on Linux. Idempotent: a second run with the
# same patch does nothing.
#   V3_SRC=<dir>   use this checkout of the pinned commit instead of fetching it
#   --force        redo the fetch even when the marker says the tree is current
#   --regen        regenerate v3/rocketsim-dump.patch from hand edits in vendor/rocketsim-<rev>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REPO=https://github.com/ZealanL/RocketSim
REV=45722876705adf5dc673aa764e80382cfe8c3312
SHORT=${REV:0:7}
# sha256 over the pristine source tree: sha256 of the sorted "file-hash  path" lines (see tree_hash)
TREE_SHA256=dd931ce2f96021c08af42ce36a697870b4abca0676b93b4e33bd4d2c0169a74e
DST="$ROOT/vendor/rocketsim-$SHORT"
PATCH="$HERE/rocketsim-dump.patch"
MARK="$DST/.dump-patch-applied"

tree_hash() { # $1 = dir; identical on Windows and Linux (no CRLF in the pinned tree, no '*' binary marker)
  (cd "$1" && find . -type f -not -path './.git/*' -not -path './target/*' -not -name .cargo-ok -not -name .dump-patch-applied -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum --binary | sed 's/^\([0-9a-f]*\) [* ]/\1  /' | sha256sum | cut -d' ' -f1)
}

# pristine <dst>: the pinned tree, from V3_SRC if set, else fetched from the repository by full commit id.
pristine() {
  mkdir -p "$1"
  if [ -n "${V3_SRC:-}" ]; then
    (cd "$V3_SRC" && tar cf - --exclude=./.git --exclude=./target --exclude=./.cargo-ok .) | (cd "$1" && tar xf -)
  else
    local tmp
    tmp="$(mktemp -d)"
    git -C "$tmp" init -q
    git -C "$tmp" config core.autocrlf false
    git -C "$tmp" fetch -q --depth 1 "$REPO" "$REV"
    git -C "$tmp" -c core.autocrlf=false archive "$REV" | tar xf - -C "$1"
    rm -rf "$tmp"
  fi
}

if [ "${1:-}" = "--regen" ]; then
  [ -d "$DST/rocketsim/src" ] || { echo "no fetched tree at $DST (run fetch_v3.sh first)" >&2; exit 1; }
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  copy() { (cd "$1" && tar cf - --exclude=./.git --exclude=./target --exclude=./.cargo-ok --exclude=./.dump-patch-applied .) \
    | (mkdir -p "$2" && cd "$2" && tar xf -); }
  pristine "$TMP/a"
  copy "$DST" "$TMP/b"
  # --no-prefix keeps the a/ and b/ directory names as the first path component, which git apply's -p1 strips
  (cd "$TMP" && git -c core.autocrlf=false diff --no-index --no-prefix --binary a b > "$TMP/new.patch") || true
  [ -s "$TMP/new.patch" ] || { echo "the tree equals the pinned commit: nothing to patch" >&2; exit 1; }
  pristine "$TMP/c"
  (cd "$TMP/c" && git -c core.autocrlf=false apply --whitespace=nowarn "$TMP/new.patch")
  diff -r "$TMP/b" "$TMP/c" > /dev/null || { echo "REFUSED: the new patch does not reproduce the tree" >&2; exit 1; }
  cp "$TMP/new.patch" "$PATCH"
  echo "wrote $(basename "$PATCH") ($(grep -c '^diff --git' "$PATCH") files); run fetch_v3.sh --force to re-fetch from it"
  exit 0
fi

[ -f "$PATCH" ] || { echo "missing $PATCH" >&2; exit 1; }
PATCH_SHA="$(sha256sum < "$PATCH" | cut -d' ' -f1)"

if [ "${1:-}" != "--force" ] && [ -f "$MARK" ] && [ -f "$DST/rocketsim/src/dump.rs" ] \
  && [ "$(cat "$MARK")" = "$TREE_SHA256 $PATCH_SHA" ]; then
  echo "vendor/rocketsim-$SHORT is current (tree $TREE_SHA256, patch $PATCH_SHA)"
  exit 0
fi

rm -rf "$DST"
pristine "$DST"

GOT="$(tree_hash "$DST")"
if [ "$GOT" != "$TREE_SHA256" ]; then
  echo "REFUSED: fetched tree sha256 $GOT differs from the expected $TREE_SHA256 (wrong commit or modified checkout)" >&2
  rm -rf "$DST"
  exit 1
fi

# git apply, run from the repository root so that the patch's paths are relative to the fetched tree
(cd "$ROOT" && git -c core.autocrlf=false apply --unsafe-paths --directory="vendor/rocketsim-$SHORT" --whitespace=nowarn "$PATCH")

printf '%s %s' "$TREE_SHA256" "$PATCH_SHA" > "$MARK"
echo "fetched rocketsim-$SHORT (tree $TREE_SHA256) and applied $(basename "$PATCH")"
