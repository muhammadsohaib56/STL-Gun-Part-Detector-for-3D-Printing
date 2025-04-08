import streamlit as st
import trimesh
import numpy as np
import io
from datetime import datetime

# --- App Config ---
st.set_page_config(page_title="3D Print Safety Scanner", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #EFF3F6; color: #1B2A41; font-family: 'Segoe UI', sans-serif; }
    .title { font-size: 3rem; font-weight: 700; color: #1B2A41; margin-bottom: 0.4rem; }
    .subtitle { font-size: 1.2rem; color: #5C8BA9; font-weight: 300; margin-bottom: 2.2rem; }
    .section { background: #FFFFFF; padding: 2rem; border-radius: 12px; box-shadow: 0 3px 6px rgba(0,0,0,0.05); border-left: 5px solid #3E92A3; }
    .css-1d391kg { background-color: #DAEAF1 !important; color: #1B2A41 !important; border-right: 1px solid #AACFD0; }
    .stButton>button { background: #3E92A3; color: #fff; border: none; border-radius: 8px; padding: 0.5rem 1.4rem; font-weight: 500; transition: background 0.3s ease; }
    .stButton>button:hover { background: #5AB3BB; }
    .stMetric { background: rgba(94, 173, 186, 0.15); padding: 1rem; border-radius: 8px; }
    .footer { text-align: center; font-size: 0.85rem; color: #888; padding: 2rem 0 1rem; margin-top: 4rem; border-top: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def is_tubular(mesh):
    """Check for tubular structures (e.g., gun barrels), even in low quality."""
    if max(mesh.extents) < 1 or len(mesh.faces) > 20000:  # Adjusted for broader range
        return False
    try:
        axis = mesh.principal_inertia_vectors[0]
        plane_origins = [mesh.centroid + axis * t for t in np.linspace(-mesh.extents[0]/2, mesh.extents[0]/2, 3)]
        plane_normals = [axis] * 3
        sections = mesh.section_multiplane(plane_origins=plane_origins, plane_normals=plane_normals)
        valid_sections = [s for s in sections if s is not None and len(s.polygons_full) > 0]
        if not valid_sections:
            return False
        areas = [s.area for s in valid_sections]
        perimeters = [s.length for s in valid_sections]
        circularities = [4 * np.pi * a / (p**2) if p > 0 else 0 for a, p in zip(areas, perimeters)]
        return any(c > 0.2 for c in circularities)  # Lowered threshold for sensitivity
    except Exception:
        return False

def has_trigger_guard(mesh):
    """Detect small, curved regions (e.g., trigger guards), even in faded meshes."""
    if mesh.is_convex or len(mesh.faces) > 20000:
        return False
    try:
        if len(mesh.vertices) > 10000:
            indices = np.random.choice(len(mesh.vertices), 10000, replace=False)
        else:
            indices = np.arange(len(mesh.vertices))
        curvature = np.zeros(len(indices))
        for idx, i in enumerate(indices):
            v = mesh.vertices[i]
            neighbors = mesh.vertex_neighbors[i]
            if len(neighbors) > 1:
                neighbor_vectors = mesh.vertices[neighbors] - v
                norms = np.linalg.norm(neighbor_vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    continue
                normalized = neighbor_vectors / norms
                dot_products = np.clip(np.dot(normalized, normalized.T), -1, 1)
                angles = np.arccos(dot_products)
                curvature[idx] = np.mean(angles)
        return np.sum(curvature > 0.25) > 2  # Lowered threshold and count for sensitivity
    except Exception:
        return False

def has_pistol_grip(mesh):
    """Detect angled protrusions (e.g., pistol grips), even with bad angles."""
    try:
        transform = mesh.principal_inertia_transform
        aligned_mesh = mesh.copy()
        aligned_mesh.apply_transform(transform)
        bounds = aligned_mesh.bounds
        z_height = bounds[1][2] - bounds[0][2]
        y_width = bounds[1][1] - bounds[0][1]
        angle_ratio = y_width / z_height if z_height > 0 else 0
        return 0.05 < angle_ratio < 4.0 and 0.5 < z_height < 400  # Widened range for detection
    except Exception:
        return False

def is_skeleton_straight(mesh):
    """Check for elongated skeletons (e.g., barrels), even in low resolution."""
    if max(mesh.extents) < 1.5 or len(mesh.faces) > 20000:
        return False
    try:
        pitch = max(mesh.extents) / 20
        if pitch < 0.05:  # Adjusted for low-res files
            pitch = 0.05
        voxel = mesh.voxelized(pitch=pitch)
        if voxel.points.size > 50000:
            return False
        centerline = voxel.medial_axis()
        if len(centerline.vertices) < 2:
            return False
        coeffs = np.polyfit(centerline.vertices[:, 0], centerline.vertices[:, 1], 1)
        residuals = np.abs(centerline.vertices[:, 1] - (coeffs[0] * centerline.vertices[:, 0] + coeffs[1]))
        return np.mean(residuals) < 1.5  # Relaxed for low-quality meshes
    except Exception:
        return False

def has_hollow_structure(mesh):
    """Detect hollow or non-watertight structures (e.g., gun chambers)."""
    try:
        if mesh.is_watertight:
            return False
        components = mesh.split(only_watertight=False)
        return len(components) > 1 or not mesh.is_watertight
    except Exception:
        return False

def preprocess_mesh(mesh):
    """Preprocess low-quality meshes for better analysis."""
    try:
        if len(mesh.faces) < 30:
            mesh = mesh.subdivide()  # Increase face count for low-res meshes
        elif len(mesh.faces) > 50000:
            mesh = mesh.simplify_quadratic_decimation(face_count=5000)  # Simplify large meshes
        return mesh
    except Exception:
        return mesh

# --- Analysis Function ---
def analyze_stl_for_gun_parts(file):
    """Analyze STL file for guns and gun parts accurately and quickly."""
    try:
        mesh = trimesh.load(file, file_type='stl', force='mesh')
        if not mesh.is_watertight:
            mesh.fill_holes()

        # Preprocess mesh for low or high quality
        mesh = preprocess_mesh(mesh)

        # Canonical orientation
        transform = mesh.principal_inertia_transform
        mesh.apply_transform(transform)

        # Basic metrics
        volume = mesh.volume
        face_count = len(mesh.faces)
        bounding_box = mesh.bounds
        dimensions = bounding_box[1] - bounding_box[0]
        aspect_ratio = max(dimensions) / min(dimensions) if min(dimensions) > 0 else 1
        convexity = mesh.is_convex
        vertex_density = len(mesh.vertices) / volume if volume > 0 else 0

        # Early exit for extremely large, non-suspicious meshes
        if face_count > 100000 and aspect_ratio < 1.2 and convexity:
            return {
                "volume": f"{volume:.2f} mm³",
                "faces": str(face_count),
                "dimensions": f"{dimensions[0]:.2f}x{dimensions[1]:.2f}x{dimensions[2]:.2f} mm",
                "aspect_ratio": f"{aspect_ratio:.2f}",
                "convexity": "Yes" if convexity else "No",
                "vertex_density": f"{vertex_density:.3f}",
                "is_tubular": "No",
                "has_trigger_guard": "No",
                "has_pistol_grip": "No",
                "straight_skeleton": "No",
                "has_hollow_structure": "No",
                "status": "Safe",
                "flagged": False,
                "mesh": mesh
            }

        # Advanced checks
        tubular = is_tubular(mesh)
        trigger_guard = has_trigger_guard(mesh)
        pistol_grip = has_pistol_grip(mesh)
        straight_skeleton = is_skeleton_straight(mesh)
        hollow = has_hollow_structure(mesh)

        # Enhanced detection logic: More sensitive to gun features
        is_gun_part_suspected = (
            (aspect_ratio > 1.2 and (tubular or straight_skeleton)) or  # Lowered aspect ratio
            (not convexity and (trigger_guard or pistol_grip)) or
            (vertex_density > 0.003 and volume < 15000) or  # Adjusted thresholds
            hollow or
            (face_count > 3 and (tubular or trigger_guard or pistol_grip))
        )

        return {
            "volume": f"{volume:.2f} mm³",
            "faces": str(face_count),
            "dimensions": f"{dimensions[0]:.2f}x{dimensions[1]:.2f}x{dimensions[2]:.2f} mm",
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "convexity": "Yes" if convexity else "No",
            "vertex_density": f"{vertex_density:.3f}",
            "is_tubular": "Yes" if tubular else "No",
            "has_trigger_guard": "Yes" if trigger_guard else "No",
            "has_pistol_grip": "Yes" if pistol_grip else "No",
            "straight_skeleton": "Yes" if straight_skeleton else "No",
            "has_hollow_structure": "Yes" if hollow else "No",
            "status": "Flagged" if is_gun_part_suspected else "Safe",
            "flagged": is_gun_part_suspected,
            "mesh": mesh
        }
    except Exception as e:
        st.error(f"Error processing STL file: {e}")
        return None

# --- Session State ---
if 'review_queue' not in st.session_state:
    st.session_state.review_queue = []

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div style="font-size: 1.8rem; font-weight: 700; color: #1B2A41; margin-bottom: 2rem;">3D Print Safety Scanner</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Home", "📤 Upload STL", "🛡️ Teacher Review"], label_visibility="collapsed", format_func=lambda x: x[2:])

# --- Header Template ---
def render_header(title, subtitle):
    st.markdown(f"""
        <div style="padding: 2rem 0;">
            <div class="title">{title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Footer Template ---
def render_footer():
    st.markdown('<div class="footer">© 2025 3D Print Safety Scanner — Powered by Streamlit</div>', unsafe_allow_html=True)

# --- Home Page ---
if page == "🏠 Home":
    render_header("🖨️ 3D Print Safety Scanner", "Ensuring safe 3D printing in school labs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("""
            ### ✨ Key Features
            - **Gun Part Detection**: Scans for prohibited items
            - **Simple Upload**: No sign-in needed
            - **Teacher Oversight**: Approve safe files
            - **Safe Printing**: Download approved STLs
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("""
            ### 🚀 How It Works
            1. Upload your STL file
            2. AI scans for gun parts
            3. Safe files downloadable; flagged files need review
        """)
        st.success("Start by uploading your file!")
        st.markdown('</div>', unsafe_allow_html=True)
    render_footer()

# --- Upload STL Page ---
elif page == "📤 Upload STL":
    render_header("📤 Upload Your STL", "Scan your 3D model for safety")
    st.markdown('<div class="section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("🧩 Upload STL File", type=["stl"], help="Supports .stl files up to 200MB")
    if uploaded_file:
        with st.spinner("Scanning for gun parts..."):
            analysis = analyze_stl_for_gun_parts(uploaded_file)
            if analysis:
                st.success(f"✅ Successfully uploaded: `{uploaded_file.name}`")
                st.markdown("### 📊 Scan Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volume", analysis["volume"])
                    st.metric("Faces", analysis["faces"])
                with col2:
                    st.metric("Dimensions", analysis["dimensions"])
                    st.metric("Aspect Ratio", analysis["aspect_ratio"])
                    st.metric("Vertex Density", analysis["vertex_density"])
                with col3:
                    st.metric("Convexity", analysis["convexity"])
                    st.metric("Tubular Shape", analysis["is_tubular"])
                    st.metric("Trigger Guard", analysis["has_trigger_guard"])
                    st.metric("Pistol Grip", analysis["has_pistol_grip"])
                    st.metric("Straight Skeleton", analysis["straight_skeleton"])
                    st.metric("Hollow Structure", analysis["has_hollow_structure"])
                    st.metric("Status", analysis["status"], "⚠️ Review" if analysis["flagged"] else "✅ Safe")

                file_entry = {
                    "name": uploaded_file.name,
                    "analysis": analysis,
                    "file_data": uploaded_file.getvalue(),
                    "timestamp": datetime.now()
                }

                if analysis["flagged"]:
                    st.warning("⚠️ Potential gun part detected! Submit for review.")
                    if st.button("Submit for Teacher Review", key="submit_review"):
                        st.session_state.review_queue.append(file_entry)
                        st.success("Submitted for review!")
                else:
                    st.success("✅ Safe to print! Available for download.")
                    if file_entry not in st.session_state.review_queue:
                        st.session_state.review_queue.append(file_entry)
    else:
        st.info("Drop an STL file to scan.")
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer()

# --- Teacher Review Page ---
elif page == "🛡️ Teacher Review":
    render_header("🛡️ Teacher Review Panel", "Review and approve STL files")
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📋 Files for Review (Latest on Top)")
    if not st.session_state.review_queue:
        st.info("No files uploaded yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.review_queue)):
            with st.expander(f"{item['name']} (Uploaded: {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})", expanded=i == 0):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volume", item["analysis"]["volume"])
                    st.metric("Faces", item["analysis"]["faces"])
                with col2:
                    st.metric("Dimensions", item["analysis"]["dimensions"])
                    st.metric("Aspect Ratio", item["analysis"]["aspect_ratio"])
                    st.metric("Vertex Density", item["analysis"]["vertex_density"])
                with col3:
                    st.metric("Convexity", item["analysis"]["convexity"])
                    st.metric("Tubular Shape", item["analysis"]["is_tubular"])
                    st.metric("Trigger Guard", item["analysis"]["has_trigger_guard"])
                    st.metric("Pistol Grip", item["analysis"]["has_pistol_grip"])
                    st.metric("Straight Skeleton", item["analysis"]["straight_skeleton"])
                    st.metric("Hollow Structure", item["analysis"]["has_hollow_structure"])
                    st.markdown("🚩 **Flagged as Potential Gun Part**" if item["analysis"]["flagged"] else "✅ **Marked Safe**")

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if not item["analysis"]["flagged"]:
                        st.download_button("⬇️ Download for Slicing", data=item["file_data"], file_name=item["name"], key=f"download_{i}")
                with col_btn2:
                    if st.button("Mark as Safe" if item["analysis"]["flagged"] else "Flag as Unsafe", key=f"flag_{i}"):
                        item["analysis"]["flagged"] = not item["analysis"]["flagged"]
                        st.experimental_rerun()
                with col_btn3:
                    if st.button("Remove", key=f"remove_{i}"):
                        original_index = len(st.session_state.review_queue) - 1 - i
                        st.session_state.review_queue.pop(original_index)
                        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer()