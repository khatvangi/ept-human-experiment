# TODO — learning-transition-inference

ordered by priority. do not drift.

## completed

- [x] 1a. delete old AGL analysis, replace with inference adapter
- [x] 1b. rewrite claim language in EXPERIMENT_DESIGN.md
- [x] 2c. grokking 60-run sweep (dose-response confirmed)
- [x] 3a. update README
- [x] 4a. web UI with exposure phase (classic Reber AGL)
- [x] 4b. backend API (/api/submit, saves JSON)
- [x] 4c. IRB consent form (USF Common Rule template)
- [x] deploy to thebeakers.com/study/

## next: before student data collection

- [ ] **pilot test with 2 students** (tomorrow)
  check: does data JSON have all fields? exposure_strings, trials,
  aha_events, confidence_trajectory, debrief. any JS errors?

- [ ] **fill IRB placeholders** in consent form
  replace [PI Name], [Department], [University], [IRB contact]
  in both /storage/thebeakers/study/index.html AND recruitment/consent_form.md

- [ ] **OSF preregistration**
  register analysis plan (S1-S4) before collecting data.
  the analysis adapter already exists — just document it on OSF.

## next: after pilot data looks clean

- [ ] **full data collection**: 60-90 students, 20-30 per condition
  links: thebeakers.com/study/?condition=easy/medium/hard

- [ ] **run analysis**: python tasks/agl/analysis/analyze_agl.py /storage/thebeakers/study/data/

## would strengthen the paper but not blocking

- [ ] **1c. hierarchical classification**
  restructure classify.py: learner/non-learner → abrupt/non-abrupt → gradual/unstable.
  ~2 hours.

- [ ] **2b. one gradual-learning dataset**
  need empirical exemplar of GRADUAL learning for topology map.
  candidates: category learning, motor skill, second-language vocab.

- [ ] **3b. SCHEMA.md**
  document canonical input/output contract of detect_transitions().

- [ ] **data dictionary**
  document all JSON fields in participant data files.

## explicitly not doing

- zebra finch (43GB raw data, not worth it)
- directory restructuring
- formal benchmark suite (only 2 real domains)
- Kramers kinetics overselling
