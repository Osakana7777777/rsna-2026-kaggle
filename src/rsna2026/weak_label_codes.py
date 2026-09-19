"""Stable compact codes used by report-derived weak labels."""

from __future__ import annotations

TARGETS = (
    "ACL",
    "MCL",
    "Medial_Meniscus",
    "Lateral_Meniscus",
    "Medial_OA",
    "Lateral_OA",
    "PF_OA",
    "Effusion",
    "Synovitis",
    "Bakers",
    "Contusion",
    "Fracture",
)

MENTION = {
    0: "not_mentioned",
    1: "explicit_positive",
    2: "explicit_negative",
    3: "equivocal",
}
CERTAINTY = {
    0: "not_applicable",
    1: "definite",
    2: "probable",
    3: "possible",
    4: "equivocal",
}
SEVERITY = {
    0: "not_stated_or_not_applicable",
    1: "none",
    2: "trace",
    3: "minimal",
    4: "small",
    5: "mild",
    6: "low_grade",
    7: "moderate",
    8: "high_grade",
    9: "severe",
    10: "large",
    11: "massive",
    12: "complete",
}
ACUITY = {
    0: "not_applicable",
    1: "not_stated",
    2: "acute",
    3: "subacute",
    4: "chronic",
    5: "remote_or_healed",
    6: "indeterminate",
}
THRESHOLD = {
    0: "insufficient_information",
    1: "clearly_below",
    2: "near_threshold",
    3: "clearly_above",
    4: "not_applicable",
}
FEATURE = {
    1: "complete_tear",
    2: "high_grade_partial_tear",
    3: "fiber_discontinuity",
    4: "low_grade_sprain",
    5: "degenerative_ligament_change",
    6: "definite_meniscal_tear",
    7: "surface_reaching_meniscal_tear",
    8: "displaced_or_abnormal_meniscal_morphology",
    9: "intrameniscal_degeneration_only",
    10: "high_grade_cartilage_loss",
    11: "full_thickness_cartilage_loss",
    12: "large_cartilage_area",
    13: "low_grade_or_focal_chondrosis",
    14: "moderate_or_large_fluid_or_cyst",
    15: "trace_or_small_fluid_or_cyst",
    16: "synovitis_or_synovial_inflammation",
    17: "synovial_thickening_or_hypertrophy",
    18: "traumatic_bone_contusion",
    19: "nonspecific_marrow_edema",
    20: "degenerative_or_stress_marrow_edema",
    21: "acute_fracture_line",
    22: "acute_cortical_break",
    23: "chronic_or_healed_fracture",
    24: "explicitly_negated",
    25: "other_threshold_relevant_feature",
    26: "acute_periligamentous_edema",
    27: "osteophyte_or_sclerosis_without_qualifying_cartilage_loss",
    28: "fracture_associated_marrow_edema",
    29: "insufficiency_or_stress_lesion",
}
REVIEW = {
    1: "severity_threshold_uncertain",
    2: "acuity_uncertain",
    3: "etiology_uncertain",
    4: "negation_uncertain",
    5: "laterality_uncertain",
    6: "conflicting_evidence",
    7: "insufficient_detail",
    8: "translation_uncertain",
    9: "other",
}
SIDE = {
    0: "unknown_or_not_stated",
    1: "left",
    2: "right",
    3: "ambiguous",
}
GLOBAL_FLAGS = {
    1: "multiple_knees_mentioned",
    2: "target_knee_uncertain",
    3: "internally_conflicting_report",
    4: "possible_language_or_negation_issue",
    5: "insufficient_or_invalid_report",
}
