import base64
from pathlib import Path
import streamlit as st
from src.ui import page_header

DEFAULT_TEAM = [
    {
        "id": "member_1",
        "name": "AbdelRhman Tamer",
        "role": "Project Lead & ML Architect",
        "bio": "Specialized in end-to-end Machine Learning systems, autonomous pipeline design, and intelligent AI decision copilots.",
        "skills": ["Machine Learning", "System Architecture", "Streamlit UI", "Pipeline Automation", "Data Analysis & Visualization"],
        "email": "abdotamer655@gmail.com",
        "github": "https://github.com/abdotamer55",
        "linkedin": "https://linkedin.com",
        "photo": "https://github.com/abdotamer55.png",
        "avatar_gradient": "linear-gradient(135deg, #6d5dfc 0%, #22d3ee 100%)",
        "initials": "AT"
    },
    {
        "id": "member_2",
        "name": "Mahmoud Talaat",
        "role": "Data Scientist & Preprocessing Lead",
        "bio": "Focuses on exploratory data analysis, automated missing value imputation, outlier handling, and data quality audits.",
        "skills": ["EDA & Statistics", "Data Cleaning", "Feature Encoding", "Data Quality"],
        "email": "",
        "github": "https://github.com/Mhmdtlat1",
        "linkedin": "https://linkedin.com",
        "photo": "https://github.com/Mhmdtlat1.png",
        "avatar_gradient": "linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%)",
        "initials": "MT"
    },
    {
        "id": "member_3",
        "name": "Ali Nagy",
        "role": "ML Engineer & Model Optimization",
        "bio": "Dedicated to model selection, hyperparameter tuning, regression/classification benchmarks, and metrics evaluation.",
        "skills": ["Scikit-Learn", "XGBoost", "Model Evaluation", "Cross-Validation"],
        "email": "",
        "github": "https://github.com/AliNagy892",
        "linkedin": "https://linkedin.com",
        "photo": "https://github.com/AliNagy892.png",
        "avatar_gradient": "linear-gradient(135deg, #3b82f6 0%, #10b981 100%)",
        "initials": "AN"
    },
    {
        "id": "member_4",
        "name": "Mostafa Tamer",
        "role": "Full-Stack UI & Visualization Engineer",
        "bio": "Crafts interactive Plotly dashboards, high-converting UI components, and seamless user experiences across workflows.",
        "skills": ["Plotly Visuals", "UI/UX Design", "Custom CSS", "Dashboard Analytics"],
        "email": "",
        "github": "https://github.com/Mostafa-tamer6",
        "linkedin": "https://linkedin.com",
        "photo": "https://github.com/Mostafa-tamer6.png",
        "avatar_gradient": "linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)",
        "initials": "MT"
    }
]

def init_team_state():
    if "team_members" not in st.session_state:
        st.session_state.team_members = [dict(m) for m in DEFAULT_TEAM]
    if "team_photos" not in st.session_state:
        st.session_state.team_photos = {}

def get_avatar_html(member):
    photo_b64 = st.session_state.team_photos.get(member["id"])
    if photo_b64:
        return f'<img src="data:image/png;base64,{photo_b64}" class="team-avatar-img" alt="{member["name"]}" />'
    elif member.get("photo"):
        return f'<img src="{member["photo"]}" class="team-avatar-img" alt="{member["name"]}" />'
    else:
        gradient = member.get("avatar_gradient", "linear-gradient(135deg, #6d5dfc, #22d3ee)")
        initials = member.get("initials", member["name"][:2].upper())
        return f'<div class="team-avatar-placeholder" style="background:{gradient};">{initials}</div>'

def render():
    init_team_state()
    
    page_header("👥 Our Team", "Meet the engineers & data scientists behind DataPilot AI")

    # Team Hero Card
    st.markdown(
        '''
        <div class="team-hero-card">
            <div class="team-hero-badge">🎓 NTI Machine Learning Graduation Project</div>
            <h2>Innovating the Future of Automated AI Systems</h2>
            <p>
                DataPilot AI was engineered collaboratively as a graduation capstone under the National Telecommunication Institute (NTI).
                Our mission is to eliminate data science friction with intelligent algorithms, interactive visual pipelines, and autonomous machine learning workflows.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Team Highlights Stats
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Team Members", f"{len(st.session_state.team_members)}")
    k2.metric("🚀 Pipeline Modules", "10+ Modules")
    k3.metric("🤖 ML Algorithms", "12+ Models")
    k4.metric("📊 Charts & Audits", "20+ Visuals")

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Member Cards Grid
    members = st.session_state.team_members
    
    # Render in 2-column or 3-column rows
    num_cols = 2 if len(members) <= 4 else 3
    for i in range(0, len(members), num_cols):
        row_members = members[i:i+num_cols]
        cols = st.columns(num_cols)
        for col, member in zip(cols, row_members):
            with col:
                avatar_html = get_avatar_html(member)
                skills_html = "".join([f'<span class="team-skill-tag">{skill}</span>' for skill in member.get("skills", [])])
                
                social_links = []
                if member.get("github"):
                    social_links.append(f'<a href="{member["github"]}" target="_blank" class="team-social-link">🐙 GitHub</a>')
                if member.get("linkedin"):
                    social_links.append(f'<a href="{member["linkedin"]}" target="_blank" class="team-social-link">💼 LinkedIn</a>')
                if member.get("email"):
                    social_links.append(f'<a href="mailto:{member["email"]}" class="team-social-link">✉️ Email</a>')
                
                socials_html = " ".join(social_links)

                st.markdown(
                    f'''
                    <div class="team-card">
                        <div class="team-card-header">
                            <div class="team-avatar-wrapper">
                                {avatar_html}
                            </div>
                            <div class="team-member-meta">
                                <h3 class="team-member-name">{member["name"]}</h3>
                                <div class="team-member-role">{member["role"]}</div>
                            </div>
                        </div>
                        <div class="team-member-bio">{member["bio"]}</div>
                        <div class="team-skills-container">
                            {skills_html}
                        </div>
                        <div class="team-socials-container">
                            {socials_html}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    # Supervisor & Institutional Acknowledgment Card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="supervisor-card">
            <div style="font-size:2rem; margin-bottom:0.4rem;">🏛️</div>
            <div style="font-weight:800; font-size:1.15rem; color:#ffffff; margin-bottom:0.3rem;">
                National Telecommunication Institute (NTI)
            </div>
            <div style="color:var(--secondary); font-weight:600; font-size:0.9rem; margin-bottom:0.6rem;">
                Specialized Machine Learning & Data Science Track
            </div>
            <p style="color:var(--text-muted); font-size:0.88rem; max-width:800px; margin:0 auto; line-height:1.6;">
                Special recognition and sincere gratitude to our project supervisors, instructors, and mentors at NTI for their invaluable guidance, support, and continuous encouragement throughout the conception and implementation of DataPilot AI.
            </p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Customization & Photo Upload Section
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ Manage Team Members & Upload Photos", expanded=False):
        st.markdown("#### ✏️ Update Team Member Details & Photos")
        st.caption("You can customize member names, roles, bios, social links, and upload profile pictures dynamically below.")

        selected_member_name = st.selectbox(
            "Select Member to Edit",
            options=[m["name"] for m in st.session_state.team_members],
            key="team_edit_select"
        )
        
        idx = next((i for i, m in enumerate(st.session_state.team_members) if m["name"] == selected_member_name), 0)
        curr_member = st.session_state.team_members[idx]

        e_col1, e_col2 = st.columns(2)
        with e_col1:
            new_name = st.text_input("Full Name", value=curr_member["name"], key=f"edit_name_{curr_member['id']}")
            new_role = st.text_input("Role / Title", value=curr_member["role"], key=f"edit_role_{curr_member['id']}")
            new_email = st.text_input("Email", value=curr_member.get("email", ""), key=f"edit_email_{curr_member['id']}")
            new_github = st.text_input("GitHub URL", value=curr_member.get("github", ""), key=f"edit_github_{curr_member['id']}")
            new_linkedin = st.text_input("LinkedIn URL", value=curr_member.get("linkedin", ""), key=f"edit_linkedin_{curr_member['id']}")
        
        with e_col2:
            new_bio = st.text_area("Bio / Tagline", value=curr_member["bio"], height=108, key=f"edit_bio_{curr_member['id']}")
            skills_str = ", ".join(curr_member.get("skills", []))
            new_skills_str = st.text_input("Skills (comma-separated)", value=skills_str, key=f"edit_skills_{curr_member['id']}")
            
            uploaded_photo = st.file_uploader(
                f"Upload Profile Photo for {curr_member['name']}",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upload_photo_{curr_member['id']}"
            )
            if uploaded_photo is not None:
                img_bytes = uploaded_photo.read()
                st.session_state.team_photos[curr_member["id"]] = base64.b64encode(img_bytes).decode("utf-8")
                st.toast(f"Photo uploaded for {curr_member['name']}! 📸", icon="✅")

        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 2])
        with btn_col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True, key="save_team_changes"):
                st.session_state.team_members[idx]["name"] = new_name
                st.session_state.team_members[idx]["role"] = new_role
                st.session_state.team_members[idx]["bio"] = new_bio
                st.session_state.team_members[idx]["email"] = new_email
                st.session_state.team_members[idx]["github"] = new_github
                st.session_state.team_members[idx]["linkedin"] = new_linkedin
                st.session_state.team_members[idx]["skills"] = [s.strip() for s in new_skills_str.split(",") if s.strip()]
                # Update initials
                parts = new_name.strip().split()
                if len(parts) >= 2:
                    st.session_state.team_members[idx]["initials"] = f"{parts[0][0]}{parts[1][0]}".upper()
                elif parts:
                    st.session_state.team_members[idx]["initials"] = parts[0][:2].upper()
                st.toast("Team member updated successfully! ✨", icon="✅")
                st.rerun()

        with btn_col2:
            if st.button("➕ Add New Member", use_container_width=True, key="add_team_member"):
                new_id = f"member_{len(st.session_state.team_members) + 1}"
                st.session_state.team_members.append({
                    "id": new_id,
                    "name": f"New Member {len(st.session_state.team_members) + 1}",
                    "role": "Data Science Specialist",
                    "bio": "Contributing to machine learning modeling, data pipelines, and analytics.",
                    "skills": ["Machine Learning", "Python", "Data Science"],
                    "email": "member@example.com",
                    "github": "https://github.com",
                    "linkedin": "https://linkedin.com",
                    "photo": None,
                    "avatar_gradient": "linear-gradient(135deg, #10b981 0%, #3b82f6 100%)",
                    "initials": "NM"
                })
                st.toast("New team member added! 🎉", icon="✅")
                st.rerun()

        with btn_col3:
            if st.button("🔄 Reset to Default Team", use_container_width=True, key="reset_team"):
                st.session_state.team_members = [dict(m) for m in DEFAULT_TEAM]
                st.session_state.team_photos = {}
                st.toast("Team members reset to defaults.", icon="🔄")
                st.rerun()
