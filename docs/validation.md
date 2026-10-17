# Release validation

The matched Gaia CNF update was checked on October 17, 2026:

* All 159,654 archived calibration/confirmation/development-test predictions replay exactly through the published adapter.
* Four new-source sectors reproduce HGB's reference and color construction exactly; prepared and new-source CLI paths agree.
* The older CNF still reproduces its 38 archived calibration predictions to absolute tolerance `1e-12`.
* Six automated tests pass, including a check that the matched model uses the HGB prior and rejects historical-cohort sources on the new-source path.
* An installed 0.1.1 wheel matches every runtime source file and runs matched CNF inference outside the repository. Dashboard generation with the new corrections passes.
* No light curves were downloaded. These checks do not validate long-baseline signal preservation.

The core extraction was checked against the preserved research implementation on October 16, 2026.

* Frozen HGB predictions replay exactly on 2,646 archived rows.
* Frozen CNF predictions replay on 38 archived calibration rows to absolute tolerance `1e-12`.
* Four cached real products pass collection, receipt/resume, physical-feature, reference and HGB correction replay. No light curves were downloaded for these checks.
* Five automated tests cover source-group exclusion, retained Gaia anchors, isolation from held-out labels, small HGB/CNF training, the two-download limit, resumption and corrupted-output detection.
* Twelve commands run from an installed wheel outside the source and research directories, including both frozen inference paths, both small training workflows and dashboard generation.
* All 775 bundled PRF files match their original acquisition checksums. Published model bundle manifests match their pinned hashes over HTTPS.

The dashboard retains the existing grid and full-cadence implementation. Its data, binary encoding and JavaScript syntax were checked. A live browser was unavailable during this release session, so an interactive visual pass was not completed. These software checks do not certify astrophysical signal preservation.

The [model manifest](../models/manifest.json) records the asset commit and checksums. [HGB preparation provenance](../models/provenance/hgb_freeze.json), [CNF selection provenance](../models/provenance/cnf_selection.json), and [application-reference audit](../models/provenance/cnf_reference_audit.csv) accompany the core source. The original model manifest retains historical report filenames; `bundle.json` defines the actual runtime assets shipped here.
