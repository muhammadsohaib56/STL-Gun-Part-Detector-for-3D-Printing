# STL-Gun-Part-Detector-for-3D-Printing
This tool scans uploaded STL files, extracts mesh metrics (volume, faces, aspect ratio, etc.), and identifies potential weapon components using shape-based heuristics. Teachers/admins can review flagged files to ensure responsible 3D printing.

Got it! Here's a **professional and beautifully formatted README description in plain text** (not in code), which you can copy and paste into your GitHub repository's `README.md` file. This version includes a clear title, project description, features, tech stack, setup instructions, usage guide, screenshots, and contribution details.

---

# 🔍 3D Print Safety Scanner | AI-Powered STL File Analyzer

A powerful AI-integrated web app designed to **analyze 3D STL files and detect firearm parts** before printing. Built with a focus on **security, digital responsibility, and user-friendly experience**, this tool is ideal for use in schools, 3D printing labs, and online STL file repositories.

---

### 🧠 **Project Highlights**
- Scans STL files for gun parts (barrels, triggers, grips, skeletons, etc.)
- Detects both **full firearms** and **individual parts** in **low or high resolution**
- Uses advanced **geometry-based heuristics** for detection
- Supports **teacher/moderator review** for flagged files
- Fast and accurate scanning with visual feedback

---

### 🛠️ **Tech Stack**
- **Frontend:** Streamlit (with custom UI and CSS styling)
- **Backend:** Python
- **3D Mesh Processing:** Trimesh, NumPy
- **Visualization & Geometry Tools:** Principal Inertia Transform, Voxelization, Section Multiplane
- **UI Enhancements:** HTML/CSS custom theming
- **Deployment:** Streamlit Cloud or Localhost

---

### 🧪 **How It Works**
1. The user uploads a `.stl` file via the web interface.
2. The model is processed to detect suspicious geometry (e.g., tubular barrels, trigger guards, grips).
3. Detailed 3D metrics are computed: volume, aspect ratio, vertex density, etc.
4. The system flags the file if patterns match known firearm shapes.
5. Flagged files can be reviewed and approved by a human moderator.

---

### 🧩 **Features**
- 📁 Upload STL files with 1 click
- 📊 Analyze structure: volume, faces, dimensions
- 🚫 Automatically flag potential gun parts
- 🧑‍🏫 Manual review interface for teachers/admins
- 💬 Clear labeling: "Safe ✅" or "Flagged 🚩"
- 🧵 Handles poor-quality or incomplete 3D models
- ⚙️ Fast, accurate, and intuitive UI

---

### 💻 **How to Run Locally**

**Step 1:** Clone the repo  
```bash
git clone https://github.com/muhammadsohaib56/stl_scanner_app.git
cd stl_scanner_app
```

**Step 2:** Install dependencies  
```bash
pip install -r requirements.txt
```

**Step 3:** Run the app  
```bash
streamlit run app.py
```

> Make sure you have Python 3.8+ and Streamlit installed.  
> STL samples can be tested via the upload section.

---

### 🤝 **Contributing**

Feel free to fork the repo and open pull requests for:
- Improving accuracy
- Adding new 3D model filters
- UI enhancements
- Bug fixes

---

### 📝 **License**
This project is licensed under the **MIT License** — feel free to use, modify, and share it responsibly.

---

### 📬 **Contact**
**Author:** Muhammad Sohaib Shoukat  
**Email:** sohaibshoukat56@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/muhammad-sohaib-shoukat-7b4064218/
**GitHub:** muhammadsohaib56

---

### 🔖 **Hashtags & Tags**
#Python #Streamlit #Trimesh #3DPrinting #STLAnalysis #GunDetection #AI #EthicalAI #DigitalSafety #SecurityInTech #OpenSource #FinalYearProject #ComputerVision #MeshProcessing #Innovation #TechForGood

---
