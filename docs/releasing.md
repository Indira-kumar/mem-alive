# Publishing `mem-alive`

The [release workflow](../.github/workflows/release.yml) publishes the latest merged `main` code to PyPI when manually dispatched. Its required `version` input becomes the version in the wheel and source distribution. The workflow changes package metadata only in the temporary runner checkout; it does not commit a version bump or create a Git tag.

## One-time setup

1. In GitHub, create an environment named `pypi` under **Settings → Environments**. Restrict it to the `main` branch and, if desired, require a maintainer's approval.
2. In the existing [PyPI `mem-alive` project](https://pypi.org/project/mem-alive/), add a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) under **Manage → Publishing** with owner `Indira-kumar`, repository `mem-alive`, workflow filename `release.yml`, and environment `pypi`. No PyPI API token is needed.

The environment name and trusted publisher fields must match the workflow exactly. These settings require access to the GitHub repository and PyPI project.

## Release

1. Merge the intended code into `main`.
2. Open **Actions → Publish to PyPI → Run workflow**. Select `main` and enter a new [PEP 440](https://packaging.python.org/en/latest/specifications/version-specifiers/) version, for example `0.1.2`.
3. The build job runs the test, lint, and type gates, changes the version in its checkout, and builds a wheel and source distribution. The `pypi` environment then gates the separate publish job, which uploads those exact artifacts using GitHub OIDC.
4. Confirm the new version on [PyPI](https://pypi.org/project/mem-alive/). PyPI will reject an already published version; use a new version for each release.

Only dispatches from `main` can publish. The release uses the current `main` HEAD at checkout time. To verify the version selection and artifacts locally without publishing, run `UV_CACHE_DIR=/tmp/mem-alive-uv-cache python evals/release_workflow.py`.
