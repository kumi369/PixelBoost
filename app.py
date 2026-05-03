from __future__ import annotations

from time import perf_counter

import streamlit as st

from pixelboost.config import APP_DESCRIPTION, APP_NAME, LARGE_IMAGE_WARNING_PIXELS, SUPPORTED_FORMATS
from pixelboost.image_utils import (
    build_download_filename,
    get_image_details,
    image_to_download_bytes,
    load_uploaded_image,
)
from pixelboost.super_resolution import (
    ModelConfigurationError,
    UpscalingError,
    get_runtime_note,
    upscale_image,
)

try:
    from streamlit_image_comparison import image_comparison
except ImportError:  # pragma: no cover - optional UI dependency
    image_comparison = None


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(18, 164, 120, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 24%),
                    linear-gradient(180deg, #09111f 0%, #101826 48%, #0b1220 100%);
            }
            .hero-card,
            .info-card {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(15, 23, 42, 0.76);
                backdrop-filter: blur(10px);
                border-radius: 22px;
                padding: 1.25rem 1.3rem;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.22);
            }
            .hero-card {
                position: relative;
                overflow: hidden;
                padding: 1.6rem 1.5rem;
                margin-bottom: 1.25rem;
            }
            .hero-card::after {
                content: "";
                position: absolute;
                inset: auto -8% -30% auto;
                width: 240px;
                height: 240px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 65%);
            }
            .hero-kicker {
                display: inline-block;
                margin-bottom: 0.75rem;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                border: 1px solid rgba(125, 211, 252, 0.2);
                background: rgba(14, 165, 233, 0.12);
                color: #bae6fd;
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.6fr) minmax(250px, 1fr);
                gap: 1rem;
                align-items: start;
            }
            .hero-metrics {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
            }
            .hero-metric,
            .feature-chip,
            .stage-card,
            .image-panel {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(255, 255, 255, 0.04);
                border-radius: 18px;
            }
            .hero-metric {
                padding: 0.95rem 1rem;
            }
            .hero-metric-label {
                color: #94a3b8;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .hero-metric-value {
                color: #f8fafc;
                font-size: 1rem;
                font-weight: 700;
                margin-top: 0.35rem;
            }
            .section-kicker {
                color: #7dd3fc;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }
            .feature-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin: 0.6rem 0 1.2rem;
            }
            .feature-chip {
                padding: 0.95rem 1rem;
            }
            .feature-chip-title {
                color: #f8fafc;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .feature-chip-copy {
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.45;
            }
            .upload-note,
            .stage-grid {
                margin: 0.75rem 0 1rem;
            }
            .upload-note {
                padding: 1rem 1.05rem;
                border-radius: 18px;
                border: 1px dashed rgba(125, 211, 252, 0.25);
                background: rgba(14, 165, 233, 0.08);
                color: #dbeafe;
            }
            .stage-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
            }
            .stage-card {
                padding: 1rem;
            }
            .stage-card-title {
                color: #f8fafc;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .stage-card-copy {
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.45;
            }
            .image-panel {
                padding: 0.9rem;
                margin-bottom: 0.9rem;
            }
            .image-panel-title {
                color: #f8fafc;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .image-panel-copy {
                color: #94a3b8;
                font-size: 0.9rem;
            }
            .metric-label {
                color: #94a3b8;
                font-size: 0.88rem;
                margin-bottom: 0.1rem;
            }
            .metric-value {
                color: #f8fafc;
                font-size: 1.05rem;
                font-weight: 600;
            }
            .success-chip {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(16, 185, 129, 0.15);
                color: #a7f3d0;
                border: 1px solid rgba(16, 185, 129, 0.24);
                font-size: 0.88rem;
                font-weight: 600;
            }
            .backend-chip {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(59, 130, 246, 0.14);
                color: #bfdbfe;
                border: 1px solid rgba(59, 130, 246, 0.25);
                font-size: 0.85rem;
                font-weight: 600;
                margin-left: 0.5rem;
            }
            @media (max-width: 900px) {
                .hero-grid,
                .hero-metrics,
                .feature-row,
                .stage-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">AI Super Resolution Studio</div>
            <div class="hero-grid">
                <div>
                    <h1 style="margin-bottom:0.45rem;font-size:2.6rem;">{APP_NAME}</h1>
                    <p style="margin:0 0 1rem;color:#cbd5e1;font-size:1.03rem;max-width:48rem;">{APP_DESCRIPTION}</p>
                    <div class="hero-metrics">
                        <div class="hero-metric">
                            <div class="hero-metric-label">Modes</div>
                            <div class="hero-metric-value">2x and 4x</div>
                        </div>
                        <div class="hero-metric">
                            <div class="hero-metric-label">Formats</div>
                            <div class="hero-metric-value">JPG, PNG, WEBP</div>
                        </div>
                        <div class="hero-metric">
                            <div class="hero-metric-label">Output</div>
                            <div class="hero-metric-value">Preview and Download</div>
                        </div>
                    </div>
                </div>
                <div class="info-card" style="margin:0;">
                    <div class="section-kicker">Workflow</div>
                    <div style="color:#f8fafc;font-weight:700;margin-bottom:0.35rem;">Upload, upscale, compare, export</div>
                    <p style="margin:0;color:#cbd5e1;line-height:1.55;">PixelBoost is tuned for fast demos and clear before/after storytelling. Drop in a low-resolution image and turn it into a cleaner, sharper asset in a single flow.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_highlights() -> None:
    st.markdown(
        """
        <div class="feature-row">
            <div class="feature-chip">
                <div class="feature-chip-title">Fast Demo Flow</div>
                <div class="feature-chip-copy">A clean upload-to-download journey designed to feel immediate and easy to present.</div>
            </div>
            <div class="feature-chip">
                <div class="feature-chip-title">Smart CPU Handling</div>
                <div class="feature-chip-copy">Larger 4x requests switch to a practical fallback path so the app stays usable on laptops.</div>
            </div>
            <div class="feature-chip">
                <div class="feature-chip-title">Clear Visual Proof</div>
                <div class="feature-chip-copy">Original preview, comparison slider, and output metadata make the enhancement easy to explain.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_details(details: dict[str, str | int], title: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="metric-label">Filename</div>
            <div class="metric-value">{details["name"]}</div>
            <div style="height:0.9rem;"></div>
            <div class="metric-label">Dimensions</div>
            <div class="metric-value">{details["width"]} x {details["height"]} px</div>
            <div style="height:0.9rem;"></div>
            <div class="metric-label">Aspect Ratio</div>
            <div class="metric-value">{details["aspect_ratio"]}</div>
            <div style="height:0.9rem;"></div>
            <div class="metric-label">Mode</div>
            <div class="metric-value">{details["mode"]}</div>
            <div style="height:0.9rem;"></div>
            <div class="metric-label">Format</div>
            <div class="metric-value">{details["format"]}</div>
            <div style="height:0.9rem;"></div>
            <div class="metric-label">File Size</div>
            <div class="metric-value">{details["size_kb"]} KB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison(original, enhanced) -> None:
    st.subheader("Before / After Comparison")
    st.caption("Focus on edges, facial details, text sharpness, and compression artifacts while comparing the two versions.")
    if image_comparison is not None:
        image_comparison(
            img1=original.convert("RGB"),
            img2=enhanced.convert("RGB"),
            label1="Original",
            label2="Upscaled",
            width=760,
        )
        st.caption("Drag the slider left and right to inspect the enhancement.")
        return

    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original", use_container_width=True)
    with right:
        st.image(enhanced, caption="Upscaled", use_container_width=True)
    st.caption("Install `streamlit-image-comparison` for an interactive slider comparison.")


def render_upload_guidance() -> None:
    st.markdown(
        """
        <div class="upload-note">
            <strong>Best results:</strong> use portraits, product shots, logos, or compressed images where clarity loss is visible. PixelBoost works best when the difference is easy to compare side by side.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_supported_formats_note() -> None:
    st.caption("Supported upload formats: `JPG`, `JPEG`, `PNG`, `WEBP`")


def render_scale_guidance(scale: int) -> None:
    if scale == 2:
        st.caption("`2x` is best for quicker enhancement passes and lighter demo images.")
    else:
        st.caption("`4x` is best when you want a stronger jump in detail and output resolution.")


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="section-kicker">Start Here</div>
        <div class="stage-grid">
            <div class="stage-card">
                <div class="stage-card-title">1. Upload an Image</div>
                <div class="stage-card-copy">Choose a JPG, PNG, or WEBP image that looks soft, compressed, or too small.</div>
            </div>
            <div class="stage-card">
                <div class="stage-card-title">2. Pick 2x or 4x</div>
                <div class="stage-card-copy">Use 2x for a quicker pass or 4x when you want a bigger jump in resolution.</div>
            </div>
            <div class="stage-card">
                <div class="stage-card-title">3. Compare and Export</div>
                <div class="stage-card-copy">Review the before/after result, inspect the size change, and download the final image.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_panel(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="image-panel">
            <div class="image-panel-title">{title}</div>
            <div class="image-panel-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upscale_estimate(width: int, height: int, scale: int) -> None:
    est_width = width * scale
    est_height = height * scale
    growth = scale * scale
    st.caption(
        f"Estimated output at `{scale}x`: `{est_width} x {est_height}` px, about `{growth}x` more pixels than the source."
    )


def render_large_image_caution(width: int, height: int, scale: int) -> None:
    if width * height >= LARGE_IMAGE_WARNING_PIXELS:
        st.warning(
            f"This image is quite large. `{scale}x` processing may take longer on CPU, especially when preview rendering is included."
        )


def render_enhancement_summary(original_details, enhanced_details, scale: int, backend_name: str, elapsed_seconds: float) -> None:
    width_gain = enhanced_details["width"] - original_details["width"]
    height_gain = enhanced_details["height"] - original_details["height"]
    pixel_multiplier = (
        (enhanced_details["width"] * enhanced_details["height"])
        // max(1, original_details["width"] * original_details["height"])
    )
    size_gain = enhanced_details["size_kb"] - original_details["size_kb"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Upscale Factor", f"{scale}x")
    col2.metric("Dimension Gain", f"+{width_gain} x +{height_gain}")
    col3.metric("Pixel Growth", f"{pixel_multiplier}x")
    col4.metric("Processing Time", f"{elapsed_seconds:.2f}s")
    col5.metric("File Size Change", f"{size_gain:+} KB")
    st.caption(f"Processed using `{backend_name}`.")
    st.caption(
        f"The source image moved from `{original_details['width']} x {original_details['height']}` to `{enhanced_details['width']} x {enhanced_details['height']}`."
    )


def main() -> None:
    inject_styles()
    render_header()
    render_feature_highlights()

    with st.sidebar:
        st.header("Settings")
        scale = st.selectbox("Upscale factor", options=[2, 4], index=1, format_func=lambda value: f"{value}x")
        st.caption("PixelBoost prefers Real-ESRGAN and falls back to a lighter pretrained OpenCV model if needed.")
        show_metadata = st.toggle("Show image metadata", value=True)
        st.markdown("---")
        st.markdown("**Performance Tips**")
        st.caption("`2x` is faster. `4x` can take longer, especially for large photos on CPU.")

    st.markdown('<div class="section-kicker">Upload</div>', unsafe_allow_html=True)
    render_upload_guidance()
    render_supported_formats_note()
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPG, JPEG, PNG, WEBP",
    )

    if uploaded_file is None:
        render_empty_state()
        return

    try:
        original_image, original_bytes = load_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    original_details = get_image_details(original_image, original_bytes, uploaded_file.name)

    preview_col, details_col = st.columns([1.5, 1], gap="large")
    with preview_col:
        render_image_panel("Original Image", "Preview the uploaded source before enhancement.")
        st.image(original_image, use_container_width=True)
    with details_col:
        if show_metadata:
            render_image_details(original_details, "Input Details")

    try:
        runtime_note = get_runtime_note(original_image, scale)
    except ModelConfigurationError as exc:
        runtime_note = None
        st.info(f"Model check note: {exc}")

    if runtime_note:
        st.warning(runtime_note)

    render_scale_guidance(scale)
    render_upscale_estimate(original_image.width, original_image.height, scale)
    render_large_image_caution(original_image.width, original_image.height, scale)

    if not st.button("Upscale Image", type="primary", use_container_width=True):
        return

    progress = st.progress(0, text="Preparing image...")
    status_box = st.empty()
    status_box.info("Preparing image for enhancement...")
    started_at = perf_counter()

    try:
        progress.progress(15, text="Loading pretrained model...")
        status_box.info("Loading the best available super-resolution backend...")
        progress.progress(45, text="Enhancing image...")
        status_box.info("Enhancing image. Larger 4x images can take longer on CPU.")
        enhanced_image, backend_name = upscale_image(original_image, scale=scale)
        progress.progress(85, text="Rendering comparison...")
        status_box.info("Rendering final preview and download output...")
        progress.progress(100, text="Enhancement complete.")
    except ModelConfigurationError as exc:
        progress.empty()
        status_box.empty()
        st.error(str(exc))
        st.info("Check the README for exact model download and placement steps.")
        return
    except UpscalingError as exc:
        progress.empty()
        status_box.empty()
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        progress.empty()
        status_box.empty()
        st.error(f"Unexpected processing failure: {exc}")
        return

    enhanced_bytes = image_to_download_bytes(enhanced_image)
    enhanced_details = get_image_details(enhanced_image, enhanced_bytes, f"pixelboost_{uploaded_file.name}")
    elapsed_seconds = perf_counter() - started_at
    status_box.success("Image enhancement finished successfully.")

    st.markdown(
        f'<p class="success-chip">Upscaling completed successfully</p><span class="backend-chip">{backend_name}</span>',
        unsafe_allow_html=True,
    )
    render_enhancement_summary(original_details, enhanced_details, scale, backend_name, elapsed_seconds)

    output_col, output_details_col = st.columns([1.5, 1], gap="large")
    with output_col:
        render_image_panel("Enhanced Image", "Review the sharpened result and inspect the visual difference.")
        st.image(enhanced_image, use_container_width=True)
    with output_details_col:
        if show_metadata:
            render_image_details(enhanced_details, "Output Details")

    render_comparison(original_image, enhanced_image)

    st.download_button(
        label="Download Enhanced Image",
        data=enhanced_bytes,
        file_name=build_download_filename(uploaded_file.name),
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
