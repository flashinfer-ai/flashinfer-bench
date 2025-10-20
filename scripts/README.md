# Scripts

## release_new_version.sh

Automates the version bump process.

### Usage

```bash
./scripts/release_new_version.sh <version>
```

**Examples:**
```bash
./scripts/release_new_version.sh v0.1.0rc1
./scripts/release_new_version.sh v0.1.0
./scripts/release_new_version.sh v0.2.0a1
```

### What it does

1. Switches to `main` branch and pulls latest changes
2. Creates a new branch `bump-<version>`
3. Updates version in `pyproject.toml`
4. Commits and pushes changes
5. Creates a PR (requires [GitHub CLI](https://cli.github.com/))

### Release workflow

1. Run the script:
   ```bash
   ./scripts/release_new_version.sh v0.1.0rc1
   ```

2. Review and merge the PR

3. Create a GitHub Release with tag `v0.1.0rc1`

4. CI/CD automatically publishes to PyPI
