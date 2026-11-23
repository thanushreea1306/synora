# SYNORA Dataset Adequacy and Supplementation Report

**Generated:** 2026-11-20  
**Scope:** Read-only investigation of whether the verified 30-class INCLUDE landmark experiment can gain stronger genuine sample support.  
**Decision:** Use the 21-class well-supported subset as the primary benchmark now; do not promote the current 30-class experiment to primary. A 30-class primary benchmark requires new genuine recordings before proceeding.

## 1. Executive Summary

- The current 30-class landmark dataset is structurally sound: 528 videos, 30,445 frames, 381 train / 51 val / 96 test, no malformed records, no split leakage, and all 32 quality audit checks passed.
- It is **not adequate as a class-balanced primary benchmark**: 9 of 30 classes fail the prior support thresholds, and the train/test video imbalance ratios are 2.67 and 4.0.
- No additional local genuine videos exist for those 9 classes. Every locally extracted official video for those classes is already selected.
- The official INCLUDE record provides no per-video signer IDs, session IDs, consent records, or demographic metadata in the local metadata. Do not claim signer-independent evaluation.
- The dataset is open under **CC BY 4.0**; the MIT license in the AI4Bharat GitHub repository covers software only.
- Minimum new recordings to make the 9 weak classes match the current well-supported class pattern (`15 train / 2 val / 4 test`) is **104 videos**.
- Recommended minimum for a more credible evaluation (`15 train / 3 val / 5 test`) is **122 new videos**.
- Recommended current design: **21 well-supported classes** using unchanged official split assignments; report the 30-class result only as exploratory unless additional recordings are collected.

## 2. Methods and Evidence Boundaries

This investigation was read-only. No data, landmarks, official split files, or source videos were changed. No videos were downloaded, recorded, duplicated, synthesized, augmented, or trained on.

Evidence sources inspected:

- Processed landmark output: `C:\Users\thanu\synora\ml\dataset\processed\synora_include30_landmarks_v1`
- Quality audit: `C:\Users\thanu\synora\ml\reports\dataset_quality_audit`
- Dataset root: `D:\SYNORA_DATA\INCLUDE`
- Official split lists: `D:\SYNORA_DATA\INCLUDE\_official_metadata\AI4Bharat_INCLUDE\train_test_paths`
- Download manifest: `D:\SYNORA_DATA\INCLUDE\_official_metadata\download_manifest.tsv`
- Zenodo record metadata: `D:\SYNORA_DATA\INCLUDE\_official_metadata\Zenodo_4010759\zenodo_record_4010759.json`
- Zenodo README and license notice: `D:\SYNORA_DATA\INCLUDE\_official_metadata\Zenodo_4010759`
- Official web evidence: Zenodo record `4010759`, Zenodo API record `4010759`, ACM DOI `10.1145/3394171.3413528`, and the AI4Bharat Hugging Face dataset card.

Hugging Face metadata was treated as documentation only. Official Zenodo split lists and the local verified manifests were treated as authoritative for sample availability.

## 3. Current Experiment Adequacy

### 3.1 Verified 30-class dataset

| Split | Videos | Frames |
|---|---:|---:|
| train | 381 | 22,004 |
| val | 51 | 3,012 |
| test | 96 | 5,429 |
| **Total** | **528** | **30,445** |

Quality audit result:

- 32 checks passed, 0 failed, 0 issues, 4 warnings.
- 528 complete videos; 0 failures, malformed records, duplicate IDs, cross-split duplicates, or split conflicts.
- Source videos remain outside the Git repository.
- No augmentation, mirroring, synthesis, frame skipping, or threshold tuning was performed.

### 3.2 Under-supported classes

The following 9 classes fail the prior thresholds (`min_train_videos=10`, `min_val_videos=2`, `min_test_videos=2`):

| Class | Train | Val | Test | Current issue |
|---|---:|---:|---:|---|
| `4. sad` | 6 | 1 | 1 | train, val, and test support |
| `5. Beautiful` | 7 | 1 | 1 | train, val, and test support |
| `17. flat` | 6 | 1 | 1 | train, val, and test support |
| `18. curved` | 6 | 1 | 1 | train, val, and test support |
| `21. tight` | 6 | 1 | 1 | train, val, and test support |
| `22. loose` | 6 | 1 | 1 | train, val, and test support |
| `29. clean` | 6 | 1 | 1 | train, val, and test support |
| `30. dirty` | 6 | 1 | 1 | train, val, and test support |
| `88. cold` | 15 | 1 | 4 | val support only |

Consequences:

- Train video imbalance is 2.67 max/min; test video imbalance is 4.0 max/min.
- Test-set macro metrics for the 1-test-video classes are single-sample estimates and are not credible per-class evidence.
- The current 30-class set is suitable only as an exploratory expansion benchmark.

### 3.3 21-class well-supported subset

Removing those 9 classes leaves 21 classes with support of 15–16 train, 2 val, and 4 test videos per class.

| Split | Videos | Frames |
|---|---:|---:|
| train | 317 | 17,880 |
| val | 42 | 2,409 |
| test | 84 | 4,695 |
| **Total** | **443** | **24,984** |

Classes retained:

`1. loud`, `2. quiet`, `3. happy`, `78. long`, `79. short`, `80. tall`, `81. wide`, `82. narrow`, `83. big large`, `84. small little`, `85. slow`, `86. fast`, `87. hot`, `89. warm`, `90. cool`, `91. new`, `92. old`, `93. young`, `94. good`, `95. bad`, `96. wet`.

This subset is narrower in vocabulary but materially fairer and more defensible for a primary benchmark.

## 4. Genuine Source Video Availability

### 4.1 Local and official availability

- Official INCLUDE record: 4,292 videos; official split lists contain 4,292 unique paths.
- Local extracted adjective videos: 728 `.MOV` files from `Adjectives_1of8.zip` through `Adjectives_7of8.zip`.
- All 728 local adjective videos match official split paths exactly; 0 local unlisted videos were found.
- The 528 selected 30-class videos are a subset of those 728 local official videos.
- The 200 local official videos not selected belong to other adjective classes, not to the 9 under-supported classes.
- `Adjectives_8of8.zip` is not locally extracted. Official paths for the missing adjective classes are `97. dry`, `98. sick`, and `99. healthy`; none of these are under-supported classes in the current 30-class set.
- For each of the 9 under-supported classes, local official count equals selected count. **Unselected local genuine videos available for those classes: 0 / 0 / 0 for train / val / test.**

Therefore, no re-selection from the local source videos can improve the 9 weak classes. Supplementation would require new genuine recordings or another appropriately licensed source with verified labels and split provenance.

### 4.2 Additional data inventory

See `additional_data_inventory.csv`. It lists per-class current counts, unused local official counts, and new recording requirements under two target designs.

## 5. Identity, Consent, and Evaluation Claims

No per-video signer ID, participant ID, recording-session ID, consent field, demographic field, or identity field was found in the local official metadata.

Observed metadata:

| Source | Contains |
|---|---|
| `download_manifest.tsv` | archive names, sizes, official/local MD5s, verification status, source URLs |
| AI4Bharat README | code/pipeline instructions, citation, label/split workflow |
| Label maps | class label to integer ID only |
| Official split lists | relative video paths only |
| Zenodo record | group-level description: videos signed by deaf students from St. Louis School for the Deaf, Adyar, Chennai |
| Hugging Face card | `parent_label`, `label`, `video_path`, `include_50`; privacy warning; no signer field |

Secondary literature reports 7 signers for INCLUDE, but the official/local metadata available here does not map videos to signers. Filename patterns such as `MVI_*` must not be treated as signer IDs.

Required claim boundary:

- Do **not** claim signer-independent evaluation.
- Do **not** infer signer identity from filenames.
- Report results as isolated-sign recognition on the official INCLUDE split, with signer overlap unverified.

If additional videos are collected, record signer/session IDs and consent information explicitly. Only build signer-disjoint train/val/test splits if signer IDs are genuinely known.

## 6. Licensing and Access

- Zenodo record `4010759`: `access_right = open`.
- Zenodo and Hugging Face both list the dataset license as **CC BY 4.0**.
- Local license notice confirms `CC BY 4.0` and `https://creativecommons.org/licenses/by/4.0/`.
- CC BY 4.0 permits reuse and redistribution with appropriate attribution.
- The MIT License in the AI4Bharat GitHub repository covers the repository software, not the video dataset license.
- Hugging Face warns that videos represent real people and requests privacy-respecting use.
- Source videos should remain outside Git; do not commit videos or derived raw archives.

## 7. Minimum Additional Samples

### Target definitions

1. **Parity target:** `15 train / 2 val / 4 test` per class. This matches the current well-supported class pattern.
2. **Recommended credible target:** `15 train / 3 val / 5 test` per class. This keeps training comparable while increasing validation and test support enough to reduce single-sample fragility.

### Parity target totals

- Current weak-class total: 64 train / 9 val / 12 test.
- Needed: +71 train / +9 val / +24 test = **104 new videos**.
- After supplementation: 485 train / 60 val / 120 test across 30 classes.

### Recommended credible target totals

- Needed: +71 train / +18 val / +33 test = **122 new videos**.
- After supplementation: 485 train / 69 val / 129 test across 30 classes.

### Per-class requirements

| Class | Current T/V/E | Parity add | Credible add |
|---|---:|---:|---:|
| `4. sad` | 6 / 1 / 1 | 13 | 15 |
| `5. Beautiful` | 7 / 1 / 1 | 12 | 14 |
| `17. flat` | 6 / 1 / 1 | 13 | 15 |
| `18. curved` | 6 / 1 / 1 | 13 | 15 |
| `21. tight` | 6 / 1 / 1 | 13 | 15 |
| `22. loose` | 6 / 1 / 1 | 13 | 15 |
| `29. clean` | 6 / 1 / 1 | 13 | 15 |
| `30. dirty` | 6 / 1 / 1 | 13 | 15 |
| `88. cold` | 15 / 1 / 4 | 1 | 3 |

Because source metadata has no signer IDs, these counts are video-count minimums only. If signer-disjoint evaluation is required, the collection plan must be redesigned around known signers and may need more videos.

## 8. Final Recommendation

### Primary benchmark: retain 21 classes

Proceed with the 21-class well-supported subset as the primary SYNORA benchmark using the unchanged official split assignments. This gives balanced, class-complete support and avoids unsupported macro-F1 claims from 1-test-video classes.

### 30-class benchmark: exploratory only

Keep the 30-class dataset as a secondary expansion experiment. Clearly label the 9 under-supported classes and avoid strong per-class or deployment claims from them.

### Gate for 30-class primary promotion

Do not promote the 30-class experiment to primary until at least **104 new genuine videos** are added, with **122** recommended. New data must be genuinely recorded, licensed, labeled, and accompanied by provenance. Do not use duplicates, synthetic samples, augmentation as sample replacement, or unlisted local files. Keep source videos outside Git.

### Claims to avoid

- No signer-independent evaluation claim.
- No claim that 30-class performance is balanced with current data.
- No claim that Hugging Face metadata rows constitute additional official videos.
- No redistribution of source videos in the repository.
