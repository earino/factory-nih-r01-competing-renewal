---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-a00c2ae77f5ae167**

[Dataset website](https://earino.github.io/factory-nih-r01-competing-renewal/versions/v-a00c2ae77f5ae167/) · [GitHub release](https://github.com/earino/factory-nih-r01-competing-renewal/releases/tag/v-a00c2ae77f5ae167) · [Hugging Face](https://huggingface.co/datasets/earino/factory-nih-r01-competing-renewal/tree/v-a00c2ae77f5ae167) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: 201d2e0b901bd8ad59d1260eb09b4b0d166abc4cf7ba476131bd46f77a7a1cbe
<!-- /dataset-factory:links -->

# Will a newly awarded NIH R01 project be continued through a competing renewal? Predicting federal research-funding continuation from award-time attributes

When NIH announces a brand-new R01 award, will that project later be continued through a competing-renewal award on the same core project number, or will the project end without one?

Derived from U.S. National Institutes of Health, Office of Extramural Research (RePORT/RePORTER ExPORTER). This is not its official product.

Prediction time: The award notice date (AWARD_NOTICE_DATE) printed on the project first-year record, the moment the new award becomes public. Features use only fields printed on that first-year record or histories built from R01 award records with AWARD_NOTICE_DATE strictly earlier than the row notice date.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6633611632117237,
    "brier": 0.09558074748502006
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.09753573340417233
  },
  "calendar_identity": {
    "auc": 0.5154745089711751,
    "brier": 0.10784755668400582
  }
}
```

## Limitations

- Retrospective snapshot, not a historical operational feed: each annual file reflects the administrative state of the records when that file was created, so later revisions (amount changes, no-cost extensions, PI moves) appear in the vintage you download; this demonstrates that the outcome is documented, not that an analyst could have queried these exact rows on the notice date.
- PROJECT_END is excluded from features because NIH extends it on renewal; TOTAL_COST may also be administratively revised, so amounts are treated as award-time values rather than audited first-year outlays.
- The outcome is partly behavioral: a renewal exists only if the investigator applied, so the label measures continuation of federal funding rather than scientific merit alone, and it co-moves with appropriations and institute paylines across fiscal years.
- Correlated observations: prolific investigators and large institutions contribute many rows, so rows are not independent draws; cluster-aware evaluation by PI and institution is advisable and the identity baselines must be reported.
- PI_ID values are not perfectly stable (accounts split or merge over time; contract and subproject cases are known to vary), so PI histories carry linkage noise.
- Direct and indirect cost fields are only populated from FY2012 onward, creating systematic missingness for the earliest training cohorts.
- Legal basis is U.S. law and territorial: 17 U.S.C. 105 removes U.S. copyright in these agency-compiled records; it is not a worldwide dedication, and no foreign rights are asserted or waived.
- Grantee-supplied text (abstracts, public health relevance statements, project titles) may carry third-party rights and is excluded: it is neither used as a feature nor redistributed. The derived release is award-level administrative metadata only, with PI identifiers pseudonymized.
- Only linked via the same agency's own files (project and, if used, PI/organization lookups); no external registry, credit, or personal data is joined.
- Retrospective snapshot, not a historical operational feed: each annual file reflects the administrative state of records when that file was created, so later revisions, amount changes, no-cost extensions and PI moves appear in the downloaded vintage.
- Historical first-year availability of each field is assumed from the current publisher data dictionary and annual files, not verified against an archived vintage.
- Actual ExPORTER publication timestamps are not established, so timing.observations is empty rather than filled with invented dates.
- PROJECT_END is excluded from features because NIH extends it on renewal; using it would leak the outcome.
- The outcome is partly behavioral: a renewal exists only if the investigator applied, so the label measures continuation of federal funding rather than scientific merit alone and co-moves with appropriations and institute paylines.
- Rows are not independent draws: prolific investigators and large institutions contribute many rows, so cluster-aware evaluation by organization and investigator history is advisable.
- PI identifier values are not perfectly stable over time, so PI histories carry linkage noise.
- Cost amounts: NIH's data dictionary describes direct and indirect cost amounts as available only for awards funded in FY2012 onward, but the current-vintage annual files used here populate the pre-FY2012 population cohorts as well (measured in the last release: no blank direct amounts and only a handful of blank indirect amounts across released rows). The descriptor now follows the measured vintage; blanks still remain missing rather than zero.
- Prior R01 history is left-censored before the first downloaded annual file, so early-cohort prior-count features may undercount older awards.
- The solicitation identifier (OPPORTUNITY_NUMBER, the FOA number) is exposed in every audited annual ExPORTER project file, but it is not shipped: the frozen feature contract for this task is the twelve award-time metadata columns, so omitting the solicitation identifier is a declared scope decision rather than a claim that the publisher files lack the column.
- No label-derived PI or organization renewal-rate feature is shipped; this is a completed check, not an unresolved item.
- The legal basis is U.S. law and territorial: 17 U.S.C. 105 removes U.S. copyright in these agency-compiled records; it is not a worldwide dedication.
- Grantee-supplied text such as abstracts, titles and public health relevance statements is excluded from features and release.
- The discovery-time provenance strings in the frozen source block (its limitations list, its construction.source_examples note and its reason summary) are preserved verbatim from source discovery and are not the operative evaluation descriptor. Two of those sentences describe an earlier snapshot: the claim that direct and indirect cost amounts are blank before FY2012 is superseded by the measured current-vintage statement above, and the phrase describing a roughly balanced label is superseded by the measured roughly eleven percent positive rate. PI identifiers are used only to derive counts and are never shipped. No built row or feature depends on those provenance sentences.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
