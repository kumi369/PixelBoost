from __future__ import annotations

from io import BytesIO

import streamlit as st

from pixelboost.config import APP_DESCRIPTION, APP_NAME, SUPPORTED_FORMATS
from pixelboost.image_utils import (
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
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(18, 164, 120, 0.12), transparent 32%),
                    radial-gradient(circle at top right, rgba(17, 94, 89, 0.12), transparent 28%),
                    linear-gradient(180deg, #0f172a 0%, #111827 45%, #0b1120 100%);
            }
            .hero-card,
            .info-card {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(15, 23, 42, 0.72);
                backdrop-filter: blur(10px);
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.22);
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin-bottom:0.4rem;">{APP_NAME}</h1>
            <p style="margin:0;color:#cbd5e1;font-size:1rem;">{APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_details(details: dict[str, str | int], title: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="metric-label">Dimensions</div>
            <div class="metric-value">{details["width"]} x {details["height"]} px</div>
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
    if image_comparison is not None:
        image_comparison(
            img1=original.convert("RGB"),
            img2=enhanced.convert("RGB"),
            label1="Original",
            label2="Upscaled",
            width=760,
        )
        return

    left, right = st.columns(2)
    with left:
        st.image(original, caption="Original", use_container_width=True)
    with right:
        st.image(enhanced, caption="Upscaled", use_container_width=True)
    st.caption("Install `streamlit-image-comparison` for an interactive slider comparison.")


def main() -> None:
    inject_styles()
    render_header()

    with st.sidebar:
        st.header("Settings")
        scale = st.selectbox("Upscale factor", options=[2, 4], index=1, format_func=lambda value: f"{value}x")
        st.caption("PixelBoost prefers Real-ESRGAN and falls back to a lighter pretrained OpenCV model if needed.")
        show_metadata = st.toggle("Show image metadata", value=True)
        st.markdown("---")
        st.markdown("**Performance Tips**")
        st.caption("`2x` is faster. `4x` can take longer, especially for large photos on CPU.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPG, JPEG, PNG, WEBP",
    )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, PNG, or WEBP image to begin.")
        return

    try:
        original_image, original_bytes = load_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    original_details = get_image_details(original_image, original_bytes, uploaded_file.name)

    preview_col, details_col = st.columns([1.5, 1], gap="large")
    with preview_col:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
    with details_col:
        if show_metadata:
            render_image_details(original_details, "Input Details")

    runtime_note = get_runtime_note(original_image, scale)
    if runtime_note:
        st.warning(runtime_note)

    if not st.button("Upscale Image", type="primary", use_container_width=True):
        return

    progress = st.progress(0, text="Preparing image...")
    status_box = st.empty()
    status_box.info("Preparing image for enhancement...")

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
    status_box.success("Image enhancement finished successfully.")

    st.markdown('<p class="success-chip">Upscaling completed successfully</p>', unsafe_allow_html=True)
    st.caption(f"Backend used: `{backend_name}`")

    output_col, output_details_col = st.columns([1.5, 1], gap="large")
    with output_col:
        st.subheader("Enhanced Image")
        st.image(enhanced_image, use_container_width=True)
    with output_details_col:
        if show_metadata:
            render_image_details(enhanced_details, "Output Details")

    render_comparison(original_image, enhanced_image)

    st.download_button(
        label="Download Enhanced Image",
        data=enhanced_bytes,
        file_name=f"pixelboost_{uploaded_file.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
