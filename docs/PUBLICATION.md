# Publication review

Publication prepared October 13, 2026, at the owner's request. Original code and
documentation are licensed under MIT, with copyright attributed to the public
GitHub handle MrM4yor. See [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md).

## Privacy checks

The pre-publication review scanned all tracked content and both existing commits
(31 unique historical blobs), including commit author/committer metadata. It found
no personal absolute paths, private email addresses, private keys, or recognized
GitHub/API credential patterns. Commits use the GitHub noreply identity selected
by the author. Untracked local demo outputs and environments are not published.

Manual inspection confirmed that bundled evidence contains aggregate experiment
statistics. There are no raw traffic exports, persistent real-vehicle identities,
individual detector-event logs, network geometry assets, or trained model files.
Public signal/intersection names identify research inputs, not the author's home
or personal travel. The private main-repository name and revision are retained
intentionally for provenance; linking it does not grant access to that repository.

`python tools/privacy_check.py --history` repeats targeted file/history checks.
CI runs it with full Git history. These checks are useful guards, not a guarantee
that every possible form of confidential material can be detected automatically.

## Release scope

- MIT applies to original project code and documentation. External dependencies
  and source datasets keep their own terms; none are relicensed by this project.
- Aggregate results retain their historical caveats, regressions, and provenance.
- Only the compact portfolio repository is made public. The main development
  repository and original artifact archives remain private/local.
- The standard-library demo verifies exports and reports preserved results;
  optional algorithm tests install numerical/modeling dependencies. Neither runs
  SUMO or regenerates the research benchmarks.

Personal research by MrM4yor, with AI-assisted implementation, debugging, review,
and documentation. No endorsement by UDOT, RESCO, Eclipse SUMO, or dependency
authors is implied.
