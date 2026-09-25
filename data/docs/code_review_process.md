# Code Review Process

## Purpose

Code review exists to catch defects before they reach production, spread
knowledge of the codebase across the team, and keep our coding standards
consistent. Every change to a shared repository goes through review before
merging.

## Opening a Pull Request

Every pull request must include a short description of what changed and
why, a link to the related ticket, and a note on how the change was
tested. PRs should be small enough to review in under 30 minutes; larger
changes should be split into a stack of smaller PRs where possible.

## Reviewer Responsibilities

At least one other engineer must approve a pull request before it merges;
changes to shared infrastructure or authentication code require two
approvals. Reviewers check for correctness, test coverage, readability,
and whether the change matches the stated intent. Reviewers should respond
to a review request within one business day, even if the response is
simply "still in progress."

## Author Responsibilities

Authors should not merge their own pull request, even if they have merge
permissions, without the required approvals. Authors are expected to
respond to review comments promptly and either make the requested change
or explain why they disagree, rather than silently dismissing feedback.

## Automated Checks

All pull requests must pass the CI pipeline, which runs the linter, the
unit test suite, and a build step, before they can be merged. A red CI
check blocks merging regardless of review approval status.

## Handling Disagreements

If an author and reviewer cannot agree after discussion in the PR, either
party can loop in a tech lead to make the final call. The goal is a quick,
respectful resolution, not "winning" the argument.

## Post-Merge

Once merged, the author is responsible for monitoring the deploy and
rolling back or hotfixing quickly if the change causes problems in
production.
