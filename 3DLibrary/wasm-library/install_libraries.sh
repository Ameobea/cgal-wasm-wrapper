#!/bin/bash
# Downloads + extracts the header-only deps into libraries/. Versions must match CMakeLists.txt,
# patches/manifest.json, and the default root in apply-patches.py.
set -e
mkdir -p libraries/downloads

fetch() {
  echo "Downloading $1"
  wget -nc -q --show-progress -O libraries/downloads/$1 $2
  echo "Extracting $1"
  tar -xf libraries/downloads/$1 -C libraries
}

fetch CGAL-6.2.tar.xz https://github.com/CGAL/cgal/releases/download/v6.2/CGAL-6.2.tar.xz
fetch boost_1_89_0.tar.gz https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
fetch eigen-3.4.0.tar.bz2 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
echo "Done"
