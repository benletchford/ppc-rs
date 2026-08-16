# Contributing

## Commit and pull request titles

Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
format:

```text
<type>(optional-scope)!: <description>
```

Allowed types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
`refactor`, `revert`, `style`, and `test`. Use `!` for a breaking change and
describe it in the commit or pull request body.

Examples:

```text
feat(mmu): add transparent translation support
fix: handle signed division overflow
refactor!: remove the legacy bus interface
```

CI validates every non-merge commit introduced by a pull request or push.
Squash-merge pull requests using the pull request title as the commit title, so
release-please evaluates the title checked by CI. The `Validate PR title` check
also fails when a title does not follow this format.

## Local checks

Run the same checks as CI before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
cargo package --locked
```

## Releases

After CI succeeds on `master`, release-please creates or updates a release pull
request from the conventional commits since the previous release. Merging that
pull request causes CI to run again, then release-please creates the Git tag and
GitHub release. The tagged crate is verified and published to crates.io using
the `CARGO_REGISTRY_TOKEN` GitHub Actions secret.

### Required GitHub setup

Create a GitHub environment named `crates.io`, then add this environment secret:

- `CARGO_REGISTRY_TOKEN`: a crates.io API token belonging to an account that is
  authorized to publish the `ppc` crate. Create and manage tokens in
  [crates.io account settings](https://crates.io/settings/tokens).

An environment secret is preferred over a repository-wide secret because only
the `publish` job declares the `crates.io` environment. Optional environment
protection rules can require approval before a release reaches crates.io.

No personal access token is required for GitHub. GitHub Actions creates
`GITHUB_TOKEN` automatically for each job; the workflow grants it the scoped
repository permissions needed by release-please and the pull-request title
check. See GitHub's [`GITHUB_TOKEN` documentation](https://docs.github.com/en/actions/concepts/security/github_token)
and Cargo's [publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).

Do not edit `CHANGELOG.md`, package versions, or
`.release-please-manifest.json` for a normal release; release-please owns those
changes.
