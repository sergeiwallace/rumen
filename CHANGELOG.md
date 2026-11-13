# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, a minor
release may add a subsystem and may change interfaces; a patch release only fixes defects and
documentation.

## [Unreleased]

### Changed

- The installation docs, the contributor guide, the CI package smoke test and the error messages
  for missing optional extras use uv commands only (`uv sync`, `uv sync --extra`, `uv add`,
  `uvx`). uv is a stated prerequisite.
- Dependabot tracks the `uv` package ecosystem.

## [0.1.0]

### Added

- First tagged release: a self-hosted engine that ingests, classifies, groups, ranks and
  digests the feeds you follow.
