"""Reproducible exploratory analysis for the RSNA knee competition data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PLANES = ["Sagittal", "Coronal", "Axial"]


def find_data_root(input_root: Path, explicit_root: Path | None) -> Path:
    """Find a mount containing the two training CSV files."""
    if explicit_root is not None:
        candidates = [explicit_root]
    else:
        candidates = [input_root / "competitions" / "rsna-knee-abnormality-detection"]
        if input_root.is_dir():
            candidates.extend(item for item in input_root.iterdir() if item.is_dir())
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "train_series.csv").is_file():
            return candidate
    raise FileNotFoundError("train.csv と train_series.csv を含む入力マウントが見つかりません。")


def save_table(table: pd.DataFrame | pd.Series, path: Path) -> None:
    """Write a table as UTF-8 CSV while preserving its index when meaningful."""
    if isinstance(table, pd.Series):
        table = table.rename(table.name or "value").to_frame()
    table.to_csv(path, encoding="utf-8-sig")


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def script_category(text: object) -> str:
    """Classify the dominant writing system of a report without language models."""
    if not isinstance(text, str) or not text.strip():
        return "Empty"
    counts = {"Latin": 0, "CJK": 0, "Cyrillic": 0, "Arabic": 0, "Other": 0}
    for character in text:
        if not character.isalpha():
            continue
        code = ord(character)
        if 0x0041 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF:
            counts["Latin"] += 1
        elif 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF:
            counts["CJK"] += 1
        elif 0x0400 <= code <= 0x052F:
            counts["Cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["Arabic"] += 1
        else:
            counts["Other"] += 1
    total = sum(counts.values())
    if total == 0:
        return "Other"
    winner = max(counts, key=counts.get)
    return winner if counts[winner] / total >= 0.7 else "Mixed"


def sorted_dicom_paths(series_dir: Path) -> list[Path]:
    """Sort slices by spatial location, then instance number, with a safe fallback."""
    records: list[dict[str, Any]] = []
    for path in series_dir.glob("*.dcm"):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True,
                                      specific_tags=["InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient"])
            instance = float(getattr(dataset, "InstanceNumber", np.nan))
            try:
                position_vector = np.asarray(dataset.ImagePositionPatient, dtype=float)
                orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
                position = float(np.dot(position_vector, np.cross(orientation[:3], orientation[3:])))
            except Exception:
                position = np.nan
        except Exception:
            instance, position = np.nan, np.nan
        records.append({"path": path, "instance": instance, "position": position})
    if not records:
        return []
    frame = pd.DataFrame(records)
    if frame["position"].notna().sum() >= max(2, len(frame) // 2):
        frame = frame.sort_values(["position", "instance"])
    elif frame["instance"].notna().sum() >= max(2, len(frame) // 2):
        frame = frame.sort_values("instance")
    else:
        frame = frame.sort_values("path", key=lambda value: value.astype(str))
    return frame["path"].tolist()


def dicom_image(dataset: Any) -> np.ndarray:
    """Convert a DICOM image into a robust 0--1 display image for EDA only."""
    pixels = dataset.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1.0)) + float(getattr(dataset, "RescaleIntercept", 0.0))
    valid = pixels[np.isfinite(pixels)]
    if not len(valid):
        return np.zeros_like(pixels)
    lower, upper = np.percentile(valid, [1, 99])
    if upper <= lower:
        lower, upper = valid.min(), valid.max()
    image = np.clip((pixels - lower) / (upper - lower), 0, 1) if upper > lower else np.zeros_like(pixels)
    return 1.0 - image if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1" else image


def save_series_panel(series_dir: Path, title: str, output: Path, n_slices: int) -> None:
    paths = sorted_dicom_paths(series_dir)
    if not paths:
        return
    indices = np.linspace(0, len(paths) - 1, min(n_slices, len(paths))).round().astype(int)
    _, axes = plt.subplots(1, len(indices), figsize=(2.4 * len(indices), 3.2))
    for axis, index in zip(np.atleast_1d(axes), indices):
        try:
            dataset = pydicom.dcmread(str(paths[index]))
            axis.imshow(dicom_image(dataset), cmap="gray")
            axis.set_title(f"{index + 1}/{len(paths)}\nInst={getattr(dataset, 'InstanceNumber', '?')}")
        except Exception as error:
            axis.text(0.5, 0.5, f"読込失敗\n{type(error).__name__}", ha="center", va="center")
        axis.axis("off")
    plt.suptitle(title)
    save_figure(output)


def sampled_metadata(series: pd.DataFrame, train_series_dir: Path) -> pd.DataFrame:
    """Read headers of sampled series only; no pixel data is decoded here."""
    rows: list[dict[str, Any]] = []
    tags = ["Rows", "Columns", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "MagneticFieldStrength", "Manufacturer", "ManufacturerModelName", "RepetitionTime", "EchoTime", "BitsStored", "PhotometricInterpretation"]
    for _, row in series.iterrows():
        directory = train_series_dir / str(row.StudyInstanceUID) / str(row.SeriesInstanceUID)
        files = list(directory.glob("*.dcm"))
        result = {key: row[key] for key in ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]}
        result["n_slices"] = len(files)
        if files:
            try:
                dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
                result.update({tag: getattr(dataset, tag, np.nan) for tag in tags})
                result["PixelSpacing"] = str(result["PixelSpacing"])
                result["TransferSyntaxUID"] = str(getattr(dataset.file_meta, "TransferSyntaxUID", ""))
            except Exception as error:
                result["error"] = str(error)
        rows.append(result)
    return pd.DataFrame(rows)


def save_study_planes(study_uid: str, train_series: pd.DataFrame, train_series_dir: Path, output: Path) -> None:
    """Save one middle slice for each anatomical plane of a study."""
    rows = train_series.loc[train_series["StudyInstanceUID"] == study_uid]
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, plane in zip(axes, PLANES):
        candidates = rows.loc[rows["Anatomical_Plane"] == plane].sort_values(["Fluid_Sensitive", "Fat_Suppression"], ascending=False)
        if candidates.empty:
            axis.text(0.5, 0.5, "シリーズなし", ha="center", va="center")
        else:
            row = candidates.iloc[0]
            paths = sorted_dicom_paths(train_series_dir / str(study_uid) / str(row.SeriesInstanceUID))
            try:
                dataset = pydicom.dcmread(str(paths[len(paths) // 2]))
                axis.imshow(dicom_image(dataset), cmap="gray")
                axis.set_title(f"{plane}\nFluid={row.Fluid_Sensitive}, FS={row.Fat_Suppression}")
            except Exception as error:
                axis.text(0.5, 0.5, f"読込失敗\n{type(error).__name__}", ha="center", va="center")
        axis.set_title(axis.get_title() or plane)
        axis.axis("off")
    plt.suptitle(f"Study: {study_uid}")
    save_figure(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RSNA Knee EDA の表・図・代表画像を保存します。")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--data-root", type=Path, help="train.csv を含む入力ディレクトリ")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/outputs/eda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-series", type=int, default=200, help="DICOMヘッダを読む系列の上限")
    parser.add_argument("--image-slices", type=int, default=7, help="代表系列パネルのスライス数")
    args = parser.parse_args(argv)
    if args.metadata_series < 1 or args.image_slices < 1:
        parser.error("--metadata-series と --image-slices は1以上にしてください。")
    output = args.output_dir
    tables, figures, images = (output / name for name in ("tables", "figures", "images"))
    for directory in (tables, figures, images):
        directory.mkdir(parents=True, exist_ok=True)
    root = find_data_root(args.input_root, args.data_root)
    train = pd.read_csv(root / "train.csv")
    train_series = pd.read_csv(root / "train_series.csv")
    missing_labels = sorted(set(LABELS) - set(train.columns))
    if missing_labels:
        raise ValueError(f"train.csv にラベル列がありません: {missing_labels}")
    series_per_study = train_series.groupby("StudyInstanceUID").size().rename("n_series")
    train["n_known_labels"] = train[LABELS].notna().sum(axis=1)
    train["report_chars"] = train["Report"].fillna("").str.len()
    train["report_words"] = train["Report"].fillna("").str.split().str.len()
    train["report_script"] = train["Report"].map(script_category)

    label_stats = pd.DataFrame([{
        "Label": label, "Labeled": int(train[label].notna().sum()),
        "Positive": int((train[label] == 1).sum()), "Negative": int((train[label] == 0).sum()),
    } for label in LABELS])
    label_stats["Positive prevalence"] = label_stats["Positive"] / label_stats["Labeled"].replace(0, np.nan)
    plane_counts = train_series["Anatomical_Plane"].value_counts(dropna=False).rename("count")
    sequence_table = pd.crosstab(train_series["Fluid_Sensitive"], train_series["Fat_Suppression"], margins=True)
    plane_sequence = train_series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"], dropna=False).size().rename("count")
    plane_pivot = pd.crosstab(train_series["Anatomical_Plane"], [train_series["Fluid_Sensitive"], train_series["Fat_Suppression"]])
    plane_presence = train_series.assign(value=1).pivot_table(index="StudyInstanceUID", columns="Anatomical_Plane", values="value", aggfunc="max", fill_value=0)
    overview = pd.DataFrame({"Metric": ["Studies", "Series", "Mean series / study", "Median series / study", "Fully labeled studies", "Completely unlabeled studies", "Studies with all three planes"], "Value": [train.StudyInstanceUID.nunique(), train_series.SeriesInstanceUID.nunique(), series_per_study.mean(), series_per_study.median(), int(train[LABELS].notna().all(axis=1).sum()), int(train[LABELS].isna().all(axis=1).sum()), float(plane_presence.reindex(columns=PLANES, fill_value=0).sum(axis=1).eq(3).mean())]})
    for name, table in {"overview": overview, "series_per_study_describe": series_per_study.describe(), "label_statistics": label_stats.sort_values("Positive prevalence", ascending=False), "known_label_counts": train.n_known_labels.value_counts().sort_index(), "plane_counts": plane_counts, "sequence_table": sequence_table, "plane_sequence": plane_sequence, "plane_by_sequence": plane_pivot, "plane_presence": plane_presence, "report_length_describe": train[["report_chars", "report_words"]].describe(), "report_script_counts": train.report_script.value_counts()}.items():
        save_table(table, tables / f"{name}.csv")

    def bar(series: pd.Series, filename: str, title: str, xlabel: str, rotation: int = 0) -> None:
        _, axis = plt.subplots(figsize=(8, 4)); axis.bar(series.index.astype(str), series.values); axis.set(title=title, xlabel=xlabel, ylabel="Count"); axis.tick_params(axis="x", rotation=rotation); save_figure(figures / filename)
    counts = train.n_known_labels.value_counts().sort_index()
    bar(counts, "known_label_counts.png", "Label availability per study", "Number of known labels")
    bar(plane_counts, "plane_counts.png", "Series by anatomical plane", "Anatomical plane")
    bar(train.report_script.value_counts(), "report_scripts.png", "Dominant writing system in reports", "Writing system", 30)
    _, axis = plt.subplots(figsize=(8, 4)); axis.hist(series_per_study, bins=np.arange(series_per_study.min(), series_per_study.max() + 2) - .5); axis.set(title="Series count per study", xlabel="Number of series per study", ylabel="Number of studies"); save_figure(figures / "series_per_study.png")
    plotted = label_stats.sort_values("Positive prevalence"); _, axis = plt.subplots(figsize=(9, 6)); axis.barh(plotted.Label, plotted["Positive prevalence"]); axis.set(title="Abnormality prevalence among labeled studies", xlabel="Positive prevalence", xlim=(0, 1)); save_figure(figures / "label_prevalence.png")
    if train[LABELS].notna().all(axis=1).sum() >= 2:
        correlation = train.loc[train[LABELS].notna().all(axis=1), LABELS].astype(float).corr(); save_table(correlation, tables / "label_correlation.csv"); _, axis = plt.subplots(figsize=(10, 9)); image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm"); axis.set_xticks(range(len(LABELS)), LABELS, rotation=90); axis.set_yticks(range(len(LABELS)), LABELS); plt.colorbar(image, ax=axis, label="Correlation"); axis.set_title("Label correlation"); save_figure(figures / "label_correlation.png")
    axis = plane_pivot.plot(kind="bar", figsize=(9, 5)); axis.set(title="Sequence characteristics by anatomical plane", ylabel="Number of series"); axis.tick_params(axis="x", rotation=0); save_figure(figures / "plane_sequence.png")
    report_lengths = train.loc[train.report_chars > 0, "report_chars"]; _, axis = plt.subplots(figsize=(8, 4)); axis.hist(report_lengths, bins=50); axis.set(title="Radiology report length", xlabel="Report length [characters]", ylabel="Number of studies"); save_figure(figures / "report_lengths.png")

    sampled = train_series.sample(n=min(args.metadata_series, len(train_series)), random_state=args.seed)
    metadata = sampled_metadata(sampled, root / "train_series")
    save_table(metadata, tables / "sampled_dicom_metadata.csv")
    _, axis = plt.subplots(figsize=(8, 4)); axis.hist(metadata.n_slices, bins=40); axis.set(title="Number of DICOM slices per series", xlabel="Slices per series", ylabel="Number of sampled series"); save_figure(figures / "dicom_slice_counts.png")
    if {"Rows", "Columns"}.issubset(metadata.columns):
        matrix = (
            metadata.dropna(subset=["Rows", "Columns"])
            .groupby(["Rows", "Columns"])
            .size()
            .sort_values(ascending=False)
            .rename("count")
        )
    else:
        matrix = pd.Series(dtype=int, name="count")
    save_table(matrix, tables / "matrix_counts.csv")
    if not matrix.empty:
        top = matrix.head(15); _, axis = plt.subplots(figsize=(10, 5)); axis.bar([f"{int(row)}×{int(column)}" for row, column in top.index], top.values); axis.set(title="Common MRI matrix sizes", xlabel="Matrix size", ylabel="Number of sampled series"); axis.tick_params(axis="x", rotation=45); save_figure(figures / "matrix_sizes.png")
    thickness_source = metadata.get("SliceThickness", pd.Series(dtype=float))
    thickness = pd.to_numeric(thickness_source, errors="coerce").dropna()
    if not thickness.empty:
        save_table(thickness.describe(), tables / "slice_thickness_describe.csv"); _, axis = plt.subplots(figsize=(8, 4)); axis.hist(thickness, bins=30); axis.set(title="Slice thickness distribution", xlabel="Slice thickness [mm]", ylabel="Number of sampled series"); save_figure(figures / "slice_thickness.png")
    manufacturers = metadata.get("Manufacturer", pd.Series(dtype=str)).replace("", np.nan).dropna().value_counts(); save_table(manufacturers, tables / "manufacturer_counts.csv")
    if not manufacturers.empty:
        bar(manufacturers, "manufacturers.png", "Scanner manufacturer", "Manufacturer", 45)
    transfer = metadata.get("TransferSyntaxUID", pd.Series(dtype=str)).replace("", np.nan).dropna().value_counts(); save_table(transfer, tables / "transfer_syntax_counts.csv")

    def duplicate_slot_rate(series: pd.DataFrame, fluid: int, fat_suppression: int) -> float:
        counts = (
            series.loc[
                (series["Fluid_Sensitive"] == fluid)
                & (series["Fat_Suppression"] == fat_suppression)
            ]
            .groupby("StudyInstanceUID")
            .size()
        )
        duplicates = counts.ge(2).reindex(plane_presence.index, fill_value=False)
        return float(duplicates.mean())

    slot_rows = []
    for plane in PLANES:
        subset = train_series[train_series.Anatomical_Plane == plane]
        slot_rows.append({
            "Plane": plane,
            "00存在率": subset.loc[
                (subset.Fluid_Sensitive == 0) & (subset.Fat_Suppression == 0),
                "StudyInstanceUID",
            ].nunique() / len(plane_presence),
            "11存在率": subset.loc[
                (subset.Fluid_Sensitive == 1) & (subset.Fat_Suppression == 1),
                "StudyInstanceUID",
            ].nunique() / len(plane_presence),
            "plane内複数series率": (
                subset.groupby("StudyInstanceUID").size()
                .reindex(plane_presence.index, fill_value=0).ge(2).mean()
            ),
            "00内複数series率": duplicate_slot_rate(subset, 0, 0),
            "11内複数series率": duplicate_slot_rate(subset, 1, 1),
        })
    save_table(pd.DataFrame(slot_rows).set_index("Plane"), tables / "plane_slot_coverage.csv")

    example = train_series.sample(1, random_state=args.seed).iloc[0]
    save_series_panel(root / "train_series" / str(example.StudyInstanceUID) / str(example.SeriesInstanceUID), f"Study: {example.StudyInstanceUID}\nSeries: {example.SeriesInstanceUID}", images / "sample_series.png", args.image_slices)
    three_plane = plane_presence.reindex(columns=PLANES, fill_value=0).sum(axis=1).loc[lambda value: value.eq(3)].index
    if len(three_plane):
        study_uid = pd.Series(three_plane).sample(1, random_state=args.seed).iloc[0]; save_study_planes(study_uid, train_series, root / "train_series", images / "three_plane_study.png")
    for label in ["ACL", "Medial Meniscus", "Effusion", "Fracture"]:
        positives = train.loc[train[label].eq(1), "StudyInstanceUID"]
        if not positives.empty:
            study_uid = positives.sample(1, random_state=args.seed).iloc[0]; save_study_planes(study_uid, train_series, root / "train_series", images / f"positive_{label.lower().replace(' ', '_')}.png")
    train.loc[train.Report.notna(), ["StudyInstanceUID", "Report"]].sample(min(5, int(train.Report.notna().sum())), random_state=args.seed).assign(Report=lambda frame: frame.Report.str.slice(0, 1000)).to_csv(tables / "sample_reports.csv", index=False, encoding="utf-8-sig")
    for filename in ("test.csv", "test_series.csv", "sample_submission.csv"):
        if (root / filename).is_file():
            save_table(pd.read_csv(root / filename).head(), tables / f"{Path(filename).stem}_head.csv")
    print(f"EDA の出力先: {output}")
    return 0
