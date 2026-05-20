from __future__ import annotations

from pathlib import Path

from PIL import Image

from config import Config


def _format_dpi(dpi_value) -> str:
    if not dpi_value:
        return "unknown"
    if isinstance(dpi_value, tuple):
        return f"{float(dpi_value[0]):.1f} x {float(dpi_value[1]):.1f}"
    return str(dpi_value)


def _dpi_values(dpi_value) -> tuple[float, float]:
    if isinstance(dpi_value, tuple) and len(dpi_value) >= 2:
        return float(dpi_value[0]), float(dpi_value[1])
    if dpi_value:
        value = float(dpi_value)
        return value, value
    return 0.0, 0.0


def main() -> None:
    config = Config()
    tiff_paths = sorted(config.plots_dir.glob("*.tiff"))
    if not tiff_paths:
        print(f"TIFF files not found in {config.plots_dir}")
        return

    for path in tiff_paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            dpi_x, dpi_y = _dpi_values(dpi)
            warnings = []
            if image.mode != "CMYK":
                warnings.append("WARNING: mode is not CMYK")
            if dpi_x < 300 or dpi_y < 300:
                warnings.append("WARNING: dpi is below 300")

            warning_text = f" | {'; '.join(warnings)}" if warnings else ""
            print(
                f"{path.name} | size={image.size[0]}x{image.size[1]} px | "
                f"dpi={_format_dpi(dpi)} | mode={image.mode}{warning_text}"
            )


if __name__ == "__main__":
    main()
