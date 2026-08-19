# Pinned `tick` environment. `tick` is the differential-test oracle for this project
# (CLAUDE.md §2), so the exact interpreter, wheel set and CPU architecture are pinned:
# fixtures committed under tests/fixtures/ must be reproducible byte-for-byte on any
# developer machine.
#
# The platform pin is deliberate. `tick` 0.8.0.2 publishes manylinux wheels for
# x86_64 only (no aarch64), so an arm64 host would otherwise have to build from
# source and could produce last-ulp differences in the reference log-likelihoods.
# Building this image on Apple Silicon works under emulation and is slow but correct.
FROM --platform=linux/amd64 python:3.13.5-slim-bookworm

# `tick` 0.8.0.2 ships manylinux_2_27/2_28 wheels, and every runtime dependency
# (numpy, scipy, pandas, scikit-learn, matplotlib) also resolves to a wheel on this
# platform, so no compiler or BLAS development package is required. Only libgomp is
# needed at run time: tick's C++ extensions are linked against OpenMP.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: \
        -r /work/requirements.txt

COPY smoke_test.py generate_fixtures.py /work/

CMD ["python", "/work/smoke_test.py"]
