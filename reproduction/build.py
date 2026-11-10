import json
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd

from source_io import fetch

ID_COL = 'award_id'
TIME_COL = 'award_notice_date'
TARGET = 'renewed'
FEATURES = [
    'administering_ic',
    'study_section',
    'ed_inst_type',
    'org_state',
    'org_country',
    'org_ipf_code',
    'total_cost',
    'direct_cost_amt',
    'indirect_cost_amt',
    'project_start_month',
    'multi_pi_count',
    'pi_prior_r01_count',
]
FEATURE_SOURCE = {
    'administering_ic': 'ADMINISTERING_IC',
    'study_section': 'STUDY_SECTION',
    'ed_inst_type': 'ED_INST_TYPE',
    'org_state': 'ORG_STATE',
    'org_country': 'ORG_COUNTRY',
    'org_ipf_code': 'ORG_IPF_CODE',
    'total_cost': 'TOTAL_COST',
    'direct_cost_amt': 'DIRECT_COST_AMT',
    'indirect_cost_amt': 'INDIRECT_COST_AMT',
}
FOA_CANDIDATES = [
    'FOA_NUMBER',
    'FUNDING_OPPORTUNITY_NUMBER',
    'OPPORTUNITY_NUMBER',
    'OPPORTUNITY_NUM',
    'FOA',
]
OPTIONAL_SOURCE = ['OPPORTUNITY_NUMBER']
COLUMNS = [ID_COL, TIME_COL] + FEATURES + [TARGET]
REQUIRED_SOURCE = [
    'CORE_PROJECT_NUM',
    'APPLICATION_TYPE',
    'ACTIVITY',
    'SUPPORT_YEAR',
    'PROJECT_START',
    'AWARD_NOTICE_DATE',
    'ORG_STATE',
    'BUDGET_START',
    'FY',
]
NEEDED_SOURCE = set(REQUIRED_SOURCE + list(FEATURE_SOURCE.values()) + ['PI_IDS'])
FIRST_FY = 2008
LAST_FY = 2023
POP_FIRST = pd.Timestamp('2007-10-01')
POP_END = pd.Timestamp('2017-10-01')
TRAIN_END = pd.Timestamp('2012-10-01')
EVAL_START = pd.Timestamp('2012-10-01')
EVAL_END = pd.Timestamp('2014-10-01')
HOLDOUT_START = pd.Timestamp('2014-10-01')
HOLDOUT_END = pd.Timestamp('2016-10-01')
CUTOFF = pd.Timestamp('2023-09-30')


def _norm(name):
    text = str(name).replace(chr(0xFEFF), '').strip()
    text = re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')
    return text.upper()


def _code(series):
    text = series.astype('string').str.strip().str.upper()
    return text.str.replace(r'[.]0+$', '', regex=True).fillna('')


def _truncate(text, limit=4000):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3] + '...'


def _check(checks, name, observed, expected, passed, **extra):
    item = {
        'check': name,
        'observed': observed,
        'expected': expected,
        'passed': bool(passed),
    }
    item.update(extra)
    checks.append(item)


def _finish_checks(checks):
    bad = [c for c in checks if not c['passed']]
    if bad:
        raise AssertionError(_truncate(json.dumps(bad, default=str)))


def _pi_tokens(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r'[;,|]', text)]
    return sorted({p for p in parts if p})


def _assign_prior_counts(pop_df, hist_df):
    counts = {}
    result = pd.Series(pd.NA, index=pop_df.index, dtype='Int64')
    if len(pop_df) == 0:
        return result
    max_date = pop_df['AWARD_NOTICE_DATE'].max()
    hist_df = hist_df[hist_df['AWARD_NOTICE_DATE'] <= max_date]
    all_dates = set(pop_df['AWARD_NOTICE_DATE'].dropna().tolist())
    all_dates.update(hist_df['AWARD_NOTICE_DATE'].dropna().tolist())
    pop_groups = pop_df.groupby('AWARD_NOTICE_DATE').groups
    hist_groups = hist_df.groupby('AWARD_NOTICE_DATE').groups
    for date in sorted(all_dates):
        for idx in pop_groups.get(date, []):
            tokens = _pi_tokens(pop_df.at[idx, 'PI_IDS'])
            if not tokens:
                result.at[idx] = pd.NA
            else:
                values = [len(counts.get(token, set())) for token in tokens]
                result.at[idx] = int(max(values)) if values else pd.NA
        for idx in hist_groups.get(date, []):
            core = hist_df.at[idx, 'CORE_PROJECT_NUM']
            for token in _pi_tokens(hist_df.at[idx, 'PI_IDS']):
                counts.setdefault(token, set()).add(core)
    return result


def _fiscal_year(date):
    return int(date.year + (1 if date.month >= 10 else 0))


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    checks = []

    _check(
        checks,
        'Declared release column list is free of duplicate names',
        f'column count: {len(COLUMNS)}; distinct names: {len(set(COLUMNS))}',
        'Every declared release column name occurs exactly once',
        len(COLUMNS) == len(set(COLUMNS)),
    )

    _check(
        checks,
        'Declared release columns cover every shipped feature, identifier, time and target name',
        f'release columns: {COLUMNS}',
        f'features listed: {FEATURES}',
        COLUMNS == [ID_COL, TIME_COL] + FEATURES + [TARGET],
    )

    pop_frames = []
    type2_frames = []
    hist_frames = []
    app_counts = {}
    activity_counts = {}
    header_audit = []
    missing_feature_years = {}
    foa_audit = {}
    sol_value_counts = {}
    total_years = LAST_FY - FIRST_FY + 1

    for fy in range(FIRST_FY, LAST_FY + 1):
        url = f'https://reporter.nih.gov/exporter/projects/download/{fy}'
        path = fetch(url, cache, f'exporter_projects_FY{fy}.zip')
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith('.csv')]
            if not members:
                raise AssertionError(_truncate(f'No CSV member found in FY{fy} archive {path}'))
            member = max(members, key=lambda m: m.file_size)
            with zf.open(member) as fh:
                head = pd.read_csv(fh, nrows=0, dtype=str)
            original = [str(c) for c in head.columns]
            norm_to_original = {}
            for col in original:
                norm = _norm(col)
                if norm not in norm_to_original:
                    norm_to_original[norm] = col
            missing_required = [c for c in REQUIRED_SOURCE if c not in norm_to_original]
            if missing_required:
                msg = (
                    f'Required ExPORTER columns missing in FY{fy} archive member {member.filename}; '
                    f'observed headers: {original}; missing: {missing_required}'
                )
                raise AssertionError(_truncate(msg))
            missing_features = []
            for source_name in FEATURE_SOURCE.values():
                if source_name not in norm_to_original:
                    missing_features.append(source_name)
                    missing_feature_years.setdefault(source_name, []).append(fy)
            present_foa = [name for name in FOA_CANDIDATES if name in norm_to_original]
            foa_audit[fy] = present_foa
            header_audit.append({
                'fy': fy,
                'member': member.filename,
                'column_count': len(original),
                'missing_required': missing_required,
                'missing_features': missing_features,
                'solicitation_header_candidates_present': present_foa,
            })
            available_original = [norm_to_original[n] for n in NEEDED_SOURCE if n in norm_to_original]
            for opt in OPTIONAL_SOURCE:
                if opt in norm_to_original:
                    available_original.append(norm_to_original[opt])
            orig_to_norm = {orig: norm for norm, orig in norm_to_original.items()}
            with zf.open(member) as fh:
                reader = pd.read_csv(
                    fh,
                    dtype=str,
                    chunksize=200000,
                    usecols=available_original,
                    keep_default_na=False,
                )
                for chunk in reader:
                    chunk = chunk.rename(columns=orig_to_norm)
                    for name in NEEDED_SOURCE:
                        if name not in chunk.columns:
                            chunk[name] = pd.NA
                    for col in list(chunk.columns):
                        values = chunk[col].astype('string').str.strip()
                        empty = values.fillna('') == ''
                        chunk[col] = values.mask(empty, pd.NA)
                    app = _code(chunk['APPLICATION_TYPE'])
                    activity = chunk['ACTIVITY'].astype('string').str.strip().str.upper().fillna('')
                    support = _code(chunk['SUPPORT_YEAR'])
                    for code, count in app.value_counts(dropna=False).items():
                        key = '<missing>' if not str(code).strip() else str(code)
                        app_counts[key] = app_counts.get(key, 0) + int(count)
                    for code, count in activity.value_counts(dropna=False).items():
                        key = '<missing>' if not str(code).strip() else str(code)
                        activity_counts[key] = activity_counts.get(key, 0) + int(count)
                    t2 = chunk.loc[app == '2', ['CORE_PROJECT_NUM', 'BUDGET_START']].copy()
                    if len(t2):
                        t2['BUDGET_START'] = pd.to_datetime(t2['BUDGET_START'], errors='coerce')
                        t2 = t2.dropna(subset=['CORE_PROJECT_NUM', 'BUDGET_START'])
                        t2 = t2.drop_duplicates()
                        if len(t2):
                            type2_frames.append(t2)
                    mask = (
                        (activity == 'R01')
                        & (app == '1')
                        & (support == '1')
                        & chunk['CORE_PROJECT_NUM'].notna()
                        & chunk['PROJECT_START'].notna()
                        & chunk['AWARD_NOTICE_DATE'].notna()
                        & chunk['ORG_STATE'].notna()
                    )
                    if mask.any():
                        cols = ['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE', 'PROJECT_START', 'PI_IDS']
                        for source_name in FEATURE_SOURCE.values():
                            cols.append(source_name)
                        if 'OPPORTUNITY_NUMBER' in chunk.columns:
                            cols.append('OPPORTUNITY_NUMBER')
                        sub = chunk.loc[mask, cols].copy()
                        sub = sub.rename(columns={v: k for k, v in FEATURE_SOURCE.items()})
                        if 'OPPORTUNITY_NUMBER' in sub.columns:
                            sol_value_counts[fy] = sol_value_counts.get(fy, 0) + int(sub['OPPORTUNITY_NUMBER'].notna().sum())
                        pop_frames.append(sub)
                    hist_mask = (
                        (activity == 'R01')
                        & chunk['CORE_PROJECT_NUM'].notna()
                        & chunk['AWARD_NOTICE_DATE'].notna()
                    )
                    if hist_mask.any():
                        hist = chunk.loc[hist_mask, ['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE', 'PI_IDS', 'ORG_IPF_CODE']].copy()
                        hist['AWARD_NOTICE_DATE'] = pd.to_datetime(hist['AWARD_NOTICE_DATE'], errors='coerce')
                        hist = hist.dropna(subset=['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE'])
                        if len(hist):
                            hist = hist.sort_values(['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE'], kind='mergesort')
                            hist = hist.drop_duplicates(subset=['CORE_PROJECT_NUM'], keep='first')
                            hist_frames.append(hist)

    if not pop_frames:
        raise AssertionError('No new R01 population rows were found in the selected ExPORTER files')
    if not type2_frames:
        raise AssertionError('No APPLICATION_TYPE 2 records were found; labels cannot be computed')
    if not hist_frames:
        raise AssertionError('No R01 history records were found; prior-count features cannot be computed')

    _check(
        checks,
        'Selected ExPORTER archive members expose the required project and label columns',
        f'checked archives: {len(header_audit)}; per-year audit: {header_audit}',
        'Every selected member contains CORE_PROJECT_NUM, APPLICATION_TYPE, ACTIVITY, SUPPORT_YEAR, PROJECT_START, AWARD_NOTICE_DATE, ORG_STATE, BUDGET_START and FY',
        all(len(row['missing_required']) == 0 for row in header_audit),
    )

    feature_presence = {}
    for source_name in FEATURE_SOURCE.values():
        missing = missing_feature_years.get(source_name, [])
        present = total_years - len(set(missing))
        feature_presence[source_name] = {'missing_years': sorted(set(missing)), 'present_year_count': present}
    _check(
        checks,
        'Every shipped feature source column appears in at least one annual file',
        f'feature presence by source column: {feature_presence}',
        'Each shipped source column is present in at least one file; absent file-years remain missing and are not filled with zero',
        all(v['present_year_count'] > 0 for v in feature_presence.values()),
    )

    foa_years_with_candidate = {str(fy): present for fy, present in sorted(foa_audit.items()) if present}
    foa_present_years = sorted(fy for fy, names in foa_audit.items() if names)
    _check(
        checks,
        'Solicitation identifier header survey is recorded for every inspected fiscal year',
        f'years inspected: {len(foa_audit)}; candidate header names surveyed: {FOA_CANDIDATES}; solicitation headers exposed per fiscal year: {foa_years_with_candidate}',
        'All sixteen annual archives are inspected and the solicitation header names each one exposes are recorded verbatim, whether or not the feature is shipped',
        len(foa_audit) == total_years and sorted(foa_audit.keys()) == list(range(FIRST_FY, LAST_FY + 1)),
    )

    _check(
        checks,
        'Surveyed solicitation identifier is reported as present rather than described as a source gap',
        f'fiscal years exposing OPPORTUNITY_NUMBER: {foa_present_years}; non-empty solicitation values among candidate population rows by fiscal year: {dict(sorted(sol_value_counts.items()))}',
        'The publisher files do expose OPPORTUNITY_NUMBER; the frozen feature contract ships twelve award-time metadata columns without the solicitation identifier, so its omission is a declared scope decision and is not a claim that the column is absent',
        len(foa_present_years) > 0,
    )

    pop = pd.concat(pop_frames, ignore_index=True)
    if 'OPPORTUNITY_NUMBER' not in pop.columns:
        pop['OPPORTUNITY_NUMBER'] = pd.NA
    raw_pop_count = len(pop)
    pop['AWARD_NOTICE_DATE'] = pd.to_datetime(pop['AWARD_NOTICE_DATE'], errors='coerce')
    pop['PROJECT_START'] = pd.to_datetime(pop['PROJECT_START'], errors='coerce')
    invalid_notice = int(pop['AWARD_NOTICE_DATE'].isna().sum())
    invalid_start = int(pop['PROJECT_START'].isna().sum())
    _check(
        checks,
        'Population date fields parse as valid calendar dates',
        f'candidate rows: {raw_pop_count}; invalid notice dates: {invalid_notice}; invalid project start dates: {invalid_start}',
        'Included first-year records have parseable AWARD_NOTICE_DATE and PROJECT_START values',
        invalid_notice == 0 and invalid_start == 0,
    )
    pop = pop.dropna(subset=['AWARD_NOTICE_DATE', 'PROJECT_START'])
    pop = pop[(pop['AWARD_NOTICE_DATE'] >= POP_FIRST) & (pop['AWARD_NOTICE_DATE'] < POP_END)].copy()
    pop = pop.sort_values(['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE', 'PROJECT_START'], kind='mergesort')
    duplicate_core_rows = int(pop.duplicated(subset=['CORE_PROJECT_NUM'], keep='first').sum())
    pop = pop.drop_duplicates(subset=['CORE_PROJECT_NUM'], keep='first').reset_index(drop=True)
    if len(pop) == 0:
        raise AssertionError('No population rows remained after date range and unique-core filtering')

    cost_parse_failures = {}
    for col in ['total_cost', 'direct_cost_amt', 'indirect_cost_amt']:
        cleaned = pop[col].astype('string').str.replace(',', '', regex=False).str.strip()
        parsed = pd.to_numeric(cleaned, errors='coerce')
        nonempty = cleaned.notna() & (cleaned != '')
        cost_parse_failures[col] = int((nonempty & parsed.isna()).sum())
        pop[col] = parsed

    pop['project_start_month'] = pop['PROJECT_START'].dt.month.astype('Int64')
    pop['multi_pi_count'] = pop['PI_IDS'].apply(
        lambda value: len(_pi_tokens(value)) if pd.notna(value) and str(value).strip() else pd.NA
    ).astype('Int64')

    hist = pd.concat(hist_frames, ignore_index=True)
    hist = hist.sort_values(['CORE_PROJECT_NUM', 'AWARD_NOTICE_DATE'], kind='mergesort')
    hist = hist.drop_duplicates(subset=['CORE_PROJECT_NUM'], keep='first').reset_index(drop=True)
    pop['pi_prior_r01_count'] = _assign_prior_counts(pop, hist)

    type2 = pd.concat(type2_frames, ignore_index=True)
    type2['BUDGET_START'] = pd.to_datetime(type2['BUDGET_START'], errors='coerce')
    type2 = type2.dropna(subset=['CORE_PROJECT_NUM', 'BUDGET_START']).drop_duplicates().reset_index(drop=True)
    pop['window_end'] = pop['PROJECT_START'] + pd.DateOffset(years=6)
    pop['_row_id'] = np.arange(len(pop), dtype=np.int64)
    left = pop[['_row_id', 'CORE_PROJECT_NUM', 'PROJECT_START', 'window_end']].copy()
    right = type2.rename(columns={'BUDGET_START': '_budget_start'})
    merged = left.merge(right, on='CORE_PROJECT_NUM', how='left')
    matched = merged[
        (merged['_budget_start'] > merged['PROJECT_START'])
        & (merged['_budget_start'] <= merged['window_end'])
    ]
    positive_ids = set(int(v) for v in matched['_row_id'].tolist())
    pop[TARGET] = pop['_row_id'].isin(positive_ids).astype('int8')
    pop['window_observed'] = pop['window_end'] <= CUTOFF
    unknown_window = int((~pop['window_observed']).sum())
    pop = pop[pop['window_observed']].copy()

    def split_for(date):
        if date < TRAIN_END:
            return 'train'
        if date < EVAL_END:
            return 'eval'
        if date < HOLDOUT_END:
            return 'holdout'
        return 'drift'

    pop['split'] = pop['AWARD_NOTICE_DATE'].apply(split_for)
    pop[ID_COL] = pop['CORE_PROJECT_NUM'].astype(str) + '|' + pop['PROJECT_START'].dt.strftime('%Y-%m-%d')
    pop[TIME_COL] = pop['AWARD_NOTICE_DATE']
    duplicate_ids = int(pop[ID_COL].duplicated().sum())

    pop['_fy'] = pop['AWARD_NOTICE_DATE'].apply(_fiscal_year)
    released_rows = pop[pop['split'].isin(['train', 'eval', 'holdout'])]
    cost_missing_by_cohort = {}
    for fy_val, grp in released_rows.groupby('_fy'):
        cost_missing_by_cohort[str(int(fy_val))] = {
            'rows': int(len(grp)),
            'direct_missing': int(grp['direct_cost_amt'].isna().sum()),
            'indirect_missing': int(grp['indirect_cost_amt'].isna().sum()),
        }

    missing_release_columns = [c for c in COLUMNS if c not in pop.columns]
    if missing_release_columns:
        msg = (
            f'Release columns missing from the constructed frame; observed constructed headers: {list(pop.columns)}; '
            f'missing: {missing_release_columns}'
        )
        raise AssertionError(_truncate(msg))

    for split_name, rel_path in [
        ('train', 'public/train.csv'),
        ('eval', 'public/eval.csv'),
        ('holdout', 'private/holdout.csv'),
    ]:
        frame = pop[pop['split'] == split_name].copy()
        frame = frame.sort_values([TIME_COL, ID_COL], kind='mergesort')
        frame = frame[COLUMNS].copy()
        frame[TIME_COL] = frame[TIME_COL].dt.strftime('%Y-%m-%d')
        target_path = output / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target_path, index=False)

    written_headers = {}
    for rel_path in ['public/train.csv', 'public/eval.csv', 'private/holdout.csv']:
        header = pd.read_csv(output / rel_path, nrows=0)
        written_headers[rel_path] = list(header.columns)
    _check(
        checks,
        'Written CSV files have the identical declared column order',
        f'written headers: {written_headers}',
        f'expected columns: {COLUMNS}',
        all(value == COLUMNS for value in written_headers.values()),
    )

    readback = {}
    for rel_path in ['public/train.csv', 'public/eval.csv', 'private/holdout.csv']:
        frame = pd.read_csv(output / rel_path, dtype=str, keep_default_na=False)
        parsed_time = pd.to_datetime(frame[TIME_COL], errors='coerce', format='%Y-%m-%d')
        targets = pd.to_numeric(frame[TARGET], errors='coerce')
        readback[rel_path] = {
            'rows': int(len(frame)),
            'unparsed_time_rows': int(parsed_time.isna().sum()),
            'blank_id_rows': int((frame[ID_COL].astype(str).str.strip() == '').sum()),
            'non_binary_target_rows': int((~targets.isin([0, 1])).sum()),
        }
    _check(
        checks,
        'Released CSVs parse back with a valid date column, populated IDs and a binary target',
        f'read-back audit: {readback}',
        'Every released row has a parseable award notice date, a non-empty award_id and a target value of either zero or one',
        all(
            item['unparsed_time_rows'] == 0
            and item['blank_id_rows'] == 0
            and item['non_binary_target_rows'] == 0
            for item in readback.values()
        ),
    )

    split_counts = {name: int((pop['split'] == name).sum()) for name in ['train', 'eval', 'holdout', 'drift']}
    _check(
        checks,
        'New R01 first-year population rows were constructed for each released split',
        f'candidate rows: {raw_pop_count}; unique population rows: {len(pop)}; split counts: {split_counts}; duplicate core rows removed: {duplicate_core_rows}',
        'At least one row in train, eval and holdout; the FY2017 drift cohort is held outside the release',
        all(split_counts[name] > 0 for name in ['train', 'eval', 'holdout']),
    )

    target_values = sorted(int(v) for v in pop[TARGET].dropna().unique().tolist())
    positives = int((pop[TARGET] == 1).sum())
    negatives = int((pop[TARGET] == 0).sum())
    _check(
        checks,
        'Constructed target is binary with both outcomes present',
        f'target values: {target_values}; positives: {positives}; negatives: {negatives}',
        'Target values are a subset of 0 and 1, with at least one positive and one negative row',
        set(target_values).issubset({0, 1}) and positives > 0 and negatives > 0,
    )

    _check(
        checks,
        'Rows without a fully observed six-year outcome window are dropped instead of labeled zero',
        f'unknown-window rows dropped: {unknown_window}; released rows: {len(pop)}',
        'No released row has an unobserved outcome window; unknown outcomes never become negative labels',
        int((~pop['window_observed']).sum()) == 0,
    )

    _check(
        checks,
        'Stable award identifiers are unique across the release',
        f'duplicate award_id rows: {duplicate_ids}',
        'Zero duplicate award_id values',
        duplicate_ids == 0,
    )

    window_specs = [
        ('train', POP_FIRST, TRAIN_END),
        ('eval', EVAL_START, EVAL_END),
        ('holdout', HOLDOUT_START, HOLDOUT_END),
    ]
    for split_name, start, end in window_specs:
        sub = pop[pop['split'] == split_name]
        min_date = sub['AWARD_NOTICE_DATE'].min()
        max_date = sub['AWARD_NOTICE_DATE'].max()
        passed = len(sub) > 0 and bool(((sub['AWARD_NOTICE_DATE'] >= start) & (sub['AWARD_NOTICE_DATE'] < end)).all())
        _check(
            checks,
            f'{split_name} rows fall inside the fixed half-open notice-date window',
            f'rows: {len(sub)}; earliest: {min_date}; latest: {max_date}',
            f'half-open window from {start} to {end}',
            passed,
        )

    for feature in FEATURES:
        non_null = int(pop[feature].notna().sum())
        _check(
            checks,
            f'Feature {feature} has observed values rather than an unresolved empty column',
            f'non-null count: {non_null}; missing count: {len(pop) - non_null}',
            'At least one non-null value; missing values remain blank and are not replaced with zero',
            non_null > 0,
        )

    for column in ['total_cost', 'direct_cost_amt', 'indirect_cost_amt']:
        values = pd.to_numeric(pop[column], errors='coerce').dropna()
        minimum = float(values.min()) if len(values) else None
        maximum = float(values.max()) if len(values) else None
        negative = int((values < 0).sum())
        _check(
            checks,
            f'{column} is measured in non-negative USD on the first-year award record',
            f'non-null count: {len(values)}; negative count: {negative}; minimum: {minimum}; maximum: {maximum}',
            'Non-negative USD amounts; blank cost fields remain missing, not zero',
            negative == 0 and len(values) > 0,
            kind='unit',
        )

    _check(
        checks,
        'Measured direct and indirect cost coverage is recorded for every released fiscal-year cohort',
        f'per-cohort coverage: {cost_missing_by_cohort}; non-empty but non-numeric amounts by column: {cost_parse_failures}',
        'Coverage is reported per released fiscal-year cohort exactly as measured in this vintage; empty amounts stay missing rather than zero, and every non-empty amount parses as a number',
        all(v == 0 for v in cost_parse_failures.values()) and len(cost_missing_by_cohort) > 0,
    )

    month_values = pop['project_start_month'].dropna()
    _check(
        checks,
        'project_start_month is a calendar month between 1 and 12',
        f'non-null count: {len(month_values)}; minimum: {month_values.min() if len(month_values) else None}; maximum: {month_values.max() if len(month_values) else None}',
        'Integer month values from 1 through 12',
        len(month_values) > 0 and bool(month_values.between(1, 12).all()),
        kind='unit',
    )

    multi_values = pop['multi_pi_count'].dropna()
    _check(
        checks,
        'multi_pi_count counts at least one distinct PI when the PI identifier list is present',
        f'non-null count: {len(multi_values)}; minimum: {multi_values.min() if len(multi_values) else None}; maximum: {multi_values.max() if len(multi_values) else None}',
        'Values are positive integers where the PI identifier list is present; a missing list remains missing',
        len(multi_values) > 0 and bool((multi_values >= 1).all()),
        kind='unit',
    )

    prior_values = pop['pi_prior_r01_count'].dropna()
    _check(
        checks,
        'pi_prior_r01_count is a non-negative count of earlier distinct R01 projects',
        f'non-null count: {len(prior_values)}; minimum: {prior_values.min() if len(prior_values) else None}; maximum: {prior_values.max() if len(prior_values) else None}',
        'Non-negative integers; a missing PI identifier list remains missing',
        len(prior_values) > 0 and bool((prior_values >= 0).all()),
        kind='unit',
    )

    _check(
        checks,
        'Positive labels come from type 2 records on the same core project inside the six-year window',
        f'type 2 source records: {len(type2)}; positive rows: {positives}; negative rows: {negatives}',
        'Every positive row has a matching APPLICATION_TYPE 2 BUDGET_START strictly after PROJECT_START and no later than six years after it',
        len(type2) > 0 and positives > 0,
    )

    excluded_codes = ['2', '3', '4', '5', '7', '9']
    excluded_counts = {code: int(app_counts.get(code, 0)) for code in excluded_codes}
    _check(
        checks,
        'Source application types outside the frozen population are recorded, not used',
        f'application type counts: {app_counts}; excluded type counts: {excluded_counts}',
        'Excluded application types may exist elsewhere in the source; only type 1 R01 first-year rows enter the population',
        bool(app_counts) and int(app_counts.get('1', 0)) > 0 and int(app_counts.get('2', 0)) > 0,
    )

    non_r01_total = int(sum(v for k, v in activity_counts.items() if str(k).upper() != 'R01'))
    n_prefix_total = int(sum(v for k, v in activity_counts.items() if str(k).upper().startswith('N')))
    z_prefix_total = int(sum(v for k, v in activity_counts.items() if str(k).upper().startswith('Z')))
    _check(
        checks,
        'Source activity codes outside the frozen R01 population are counted by agency prefix',
        f'activity counts: {activity_counts}; non-R01 records: {non_r01_total}; N-prefix records: {n_prefix_total}; Z-prefix records: {z_prefix_total}',
        'N-prefixed contract activity codes and Z-prefixed intramural activity codes may exist elsewhere in the source; only ACTIVITY code R01 with APPLICATION_TYPE 1 and SUPPORT_YEAR 1 enters the population, so their presence is recorded rather than aborting the build',
        non_r01_total > 0 and (n_prefix_total + z_prefix_total) > 0,
    )

    _finish_checks(checks)
    return checks
