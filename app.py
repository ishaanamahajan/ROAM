"""Roam — learn a traveler's taste from a handful of visual choices.

Run with: ``streamlit run app.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from roam.data import DESTINATIONS, FEATURES, feature_matrix
from roam.model import (
    aggregate_group_scores,
    contribution_explanation,
    fit_preference_model,
    ranked_recommendations,
    select_next_pair,
)
from roam.profiles import demo_profiles, profile_from_json, profile_to_json


st.set_page_config(page_title="Roam · Find your somewhere", page_icon="🧭", layout="wide")

ROOT = Path(__file__).resolve().parent
MATRIX = feature_matrix()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#14221c; --fern:#1f5c45; --mint:#dcebe3; --paper:#f7f4ed; --coral:#e56b4b; }
        .stApp { background: var(--paper); color: var(--ink); }
        html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
        h1, h2, h3 { font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif !important; color:var(--ink); letter-spacing:-.025em; }
        h1 { font-size:clamp(2.8rem,6vw,5.2rem) !important; line-height:.95 !important; }
        [data-testid="stSidebar"] { background:#173f32; }
        [data-testid="stSidebar"] * { color:#f7f4ed !important; }
        [data-testid="stSidebar"] h1 { font-size:1.75rem !important; line-height:1.1 !important; letter-spacing:.04em; white-space:nowrap; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.18rem 0; }
        [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.2); }
        .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.76rem; font-weight:700; color:var(--fern); }
        .hero-copy { font-size:1.18rem; line-height:1.6; color:#4b5b53; max-width:680px; }
        .pill { display:inline-block; padding:.34rem .68rem; border:1px solid #b8cdc1; border-radius:999px; color:#315d49; font-size:.78rem; margin:.1rem .2rem .1rem 0; }
        .metric-card { background:#fff; border:1px solid #dce4df; border-radius:16px; padding:1rem 1.1rem; min-height:108px; }
        .metric-number { font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif; font-size:2rem; color:var(--fern); }
        .metric-label { color:#66746d; font-size:.84rem; }
        div[data-testid="stButton"] > button { background:#fff; color:#1f5c45; border-radius:999px; border:1px solid #1f5c45; font-weight:700; min-height:2.8rem; }
        div[data-testid="stButton"] > button * { color:inherit !important; }
        div[data-testid="stButton"] > button[kind="primary"] { background:#1f5c45; color:#fff; }
        [data-testid="stDownloadButton"] > button { background:#fff !important; color:#1f5c45 !important; border:1px solid #1f5c45 !important; border-radius:999px; font-weight:700; min-height:2.8rem; }
        [data-testid="stDownloadButton"] > button * { color:#1f5c45 !important; }
        [data-testid="stImage"] img { border-radius:18px; box-shadow:0 10px 35px rgba(20,34,28,.12); }
        div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.72); border-color:#d7e0db !important; border-radius:18px; }
        .why { color:#5f6d66; font-size:.91rem; line-height:1.45; min-height:2.7rem; }
        .footer-note { margin-top:4rem; padding-top:1rem; border-top:1px solid #d8ded9; color:#7a847f; font-size:.82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "comparisons": [],
        "shown_pairs": [],
        "current_pair": None,
        "profile_name": "My profile",
        "saved_profiles": {},
        "import_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_profile():
    return fit_preference_model(st.session_state.comparisons, MATRIX)


def seen_destinations() -> set[int]:
    return {index for comparison in st.session_state.comparisons for index in comparison}


def record_choice(winner: int, loser: int, pair: tuple[int, int]) -> None:
    st.session_state.comparisons.append((winner, loser))
    st.session_state.shown_pairs.append(tuple(sorted(pair)))
    st.session_state.current_pair = None


def destination_image(index: int) -> None:
    destination = DESTINATIONS[index]
    st.image(str(destination.image_path), width="stretch")


def destination_copy(index: int, compact: bool = False) -> None:
    destination = DESTINATIONS[index]
    st.markdown(f"### {destination.name}")
    st.caption(f"{destination.country} · {destination.region}")
    if not compact:
        st.markdown(f'<div class="why">{destination.blurb}</div>', unsafe_allow_html=True)
        tags = sorted(
            zip(FEATURES, destination.features), key=lambda item: item[1], reverse=True
        )[:3]
        st.markdown("".join(f'<span class="pill">{name}</span>' for name, _ in tags), unsafe_allow_html=True)


def hero() -> None:
    left, right = st.columns([1.25, .75], vertical_alignment="center")
    with left:
        st.markdown('<div class="eyebrow">Personal travel, learned by choice</div>', unsafe_allow_html=True)
        st.title("Find your\nsomewhere.")
        st.markdown(
            '<div class="hero-copy">Skip the filters. Pick the place that pulls you in, and Roam will learn what makes a destination feel right for you.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.image(str(DESTINATIONS[10].image_path), width="stretch")


def discover_page() -> None:
    hero()
    comparisons = len(st.session_state.comparisons)
    st.markdown("---")
    top_left, top_right = st.columns([.72, .28], vertical_alignment="center")
    with top_left:
        st.markdown('<div class="eyebrow">Taste check</div>', unsafe_allow_html=True)
        st.header("Where would you rather go?")
        st.caption("Go with your instinct — there are no wrong answers.")
    with top_right:
        st.progress(min(comparisons / 7, 1.0), text=f"{comparisons} of 7 choices · strong start")

    profile = current_profile()
    if st.session_state.current_pair is None:
        st.session_state.current_pair = select_next_pair(
            profile, MATRIX, st.session_state.shown_pairs, seen_destinations()
        )
    left_index, right_index = st.session_state.current_pair
    left, spacer, right = st.columns([1, .08, 1])
    with left:
        destination_image(left_index)
        destination_copy(left_index)
        if st.button(f"Choose {DESTINATIONS[left_index].name}", key=f"left-{comparisons}", type="primary", width="stretch"):
            record_choice(left_index, right_index, (left_index, right_index))
            st.rerun()
    with spacer:
        st.markdown("<div style='height:10rem'></div><div style='text-align:center;font-weight:700;color:#7c8c84'>OR</div>", unsafe_allow_html=True)
    with right:
        destination_image(right_index)
        destination_copy(right_index)
        if st.button(f"Choose {DESTINATIONS[right_index].name}", key=f"right-{comparisons}", type="primary", width="stretch"):
            record_choice(right_index, left_index, (left_index, right_index))
            st.rerun()

    action_left, action_mid, action_right = st.columns([1, 1, 2])
    with action_left:
        if comparisons and st.button("↶ Undo last choice", width="stretch"):
            st.session_state.comparisons.pop()
            if st.session_state.shown_pairs:
                st.session_state.shown_pairs.pop()
            st.session_state.current_pair = None
            st.rerun()
    with action_mid:
        if comparisons and st.button("Start over", width="stretch"):
            st.session_state.comparisons = []
            st.session_state.shown_pairs = []
            st.session_state.current_pair = None
            st.rerun()
    with action_right:
        if comparisons >= 3:
            st.info("Your first recommendations are ready. Keep choosing to sharpen them, or open **My Taste**.")


def recommendation_card(index: int, match: float, profile, key: str) -> None:
    destination = DESTINATIONS[index]
    with st.container(border=True):
        destination_image(index)
        col_name, col_match = st.columns([.72, .28])
        with col_name:
            st.markdown(f"### {destination.name}")
            st.caption(f"{destination.country} · {destination.region}")
        with col_match:
            st.metric("Taste match", f"{match:.0f}%")
        reasons = contribution_explanation(profile, MATRIX[index], FEATURES)
        reason_text = " + ".join(reasons) if reasons else destination.best_for
        st.markdown(f"**Why it fits:** {reason_text}")
        st.markdown(f'<div class="why">{destination.blurb}</div>', unsafe_allow_html=True)


def taste_page() -> None:
    st.markdown('<div class="eyebrow">Your travel DNA</div>', unsafe_allow_html=True)
    st.title("Your taste, decoded.")
    profile = current_profile()
    if not profile.comparisons:
        st.info("Make a few choices in **Discover** and your preference map will appear here.")
        return

    metric_columns = st.columns(3)
    metrics = [
        (str(profile.comparisons), "pairwise choices"),
        (f"{profile.confidence * 100:.0f}%", "profile strength"),
        (str(len(DESTINATIONS) - len(seen_destinations())), "unseen places ranked"),
    ]
    for column, (number, label) in zip(metric_columns, metrics):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-number">{number}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.header("What Roam has learned")
    order = np.argsort(np.abs(profile.weights))[::-1]
    chart_left, chart_right = st.columns([.62, .38])
    with chart_left:
        for feature_index in order[:6]:
            weight = profile.weights[feature_index]
            sentiment = "drawn to" if weight >= 0 else "less focused on"
            st.markdown(f"**{FEATURES[feature_index]}** · {sentiment}")
            normalized = int(np.clip(50 + weight * 24, 2, 98))
            st.progress(normalized / 100)
    with chart_right:
        st.info(
            "Roam learns tradeoffs, not a checklist. A negative bar doesn't mean dislike — it means that quality mattered less in the choices you made."
        )
        name = st.text_input("Profile name", value=st.session_state.profile_name)
        st.session_state.profile_name = name
        if st.button("Save for group mode", width="stretch"):
            st.session_state.saved_profiles[name.strip() or "My profile"] = profile
            st.success("Profile saved in this session.")
        st.download_button(
            "Download shareable profile",
            data=profile_to_json(name, profile),
            file_name="roam-profile.json",
            mime="application/json",
            width="stretch",
        )

    st.header("Places picked for you")
    st.caption("These are destinations you haven't seen in the taste check, ranked by learned fit.")
    recommendations = ranked_recommendations(profile, MATRIX, seen_destinations(), limit=6)
    rows = [recommendations[:3], recommendations[3:6]]
    for row_number, row in enumerate(rows):
        columns = st.columns(3)
        for column, (index, match) in zip(columns, row):
            with column:
                recommendation_card(index, match, profile, f"rec-{row_number}-{index}")


def group_page() -> None:
    st.markdown('<div class="eyebrow">Better together</div>', unsafe_allow_html=True)
    st.title("Find the group’s happy place.")
    st.markdown(
        '<div class="hero-copy">Combine taste profiles and balance the crowd favorite against the place that leaves nobody behind.</div>',
        unsafe_allow_html=True,
    )

    library = demo_profiles()
    library.update(st.session_state.saved_profiles)
    profile = current_profile()
    if profile.comparisons:
        library[st.session_state.profile_name.strip() or "My profile"] = profile

    setup, import_col = st.columns([.62, .38])
    with setup:
        members = st.multiselect("Who's traveling?", options=list(library), default=[])
        st.caption("Roam gives each traveler equal influence by normalizing their destination scores before averaging them.")
    with import_col:
        st.markdown("**Add a friend's profile**")
        uploaded = st.file_uploader("Upload their Roam JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            signature = (uploaded.name, uploaded.size)
            if st.session_state.import_message != signature:
                try:
                    imported_name, imported_profile = profile_from_json(uploaded.getvalue().decode("utf-8"))
                    st.session_state.saved_profiles[imported_name] = imported_profile
                    st.session_state.import_message = signature
                    st.success(f"Added {imported_name}. They will appear after the page refreshes.")
                    st.rerun()
                except (ValueError, UnicodeDecodeError) as error:
                    st.error(str(error))
        st.caption("The three named examples are synthetic demo profiles and are clearly separated from your learned profile.")

    if len(members) < 2:
        st.warning("Choose at least two travelers to make a group recommendation.")
        return

    profiles = [library[name] for name in members]
    group_scores, disagreement = aggregate_group_scores(profiles, MATRIX)
    scale = max(float(group_scores.std()), .35)
    matches = 100 / (1 + np.exp(-group_scores / scale))
    ranked = sorted(range(len(DESTINATIONS)), key=lambda index: (-group_scores[index], index))[:6]

    st.header("The shortlist")
    st.caption(f"Average preference across {len(members)} equally weighted travelers")
    for rank, index in enumerate(ranked, start=1):
        destination = DESTINATIONS[index]
        with st.container(border=True):
            image_col, copy_col, score_col = st.columns([.30, .48, .22], vertical_alignment="center")
            with image_col:
                destination_image(index)
            with copy_col:
                st.markdown(f"### {rank}. {destination.name}, {destination.country}")
                st.write(destination.blurb)
                agreement_label = "easy consensus" if disagreement[index] < .55 else "some tradeoffs"
                st.caption(f"{destination.best_for.title()} · {agreement_label}")
            with score_col:
                st.metric("Group match", f"{matches[index]:.0f}%")
                st.metric("Disagreement", f"{disagreement[index]:.2f}", help="Lower is better; 0 means identical enthusiasm.")


def about_page() -> None:
    st.markdown('<div class="eyebrow">Inside Roam</div>', unsafe_allow_html=True)
    st.title("Small choices. Useful intelligence.")
    intro, facts = st.columns([.62, .38])
    with intro:
        st.markdown(
            """
            Roam is a transparent prototype of preference learning. Each destination has a compact
            visual-semantic embedding across ten travel qualities. When you choose A over B, the model
            learns which differences best explain that decision.

            Under the hood, Roam fits a regularized **Bradley–Terry logistic utility model**. The next
            question is selected actively: it favors comparisons that are uncertain, close, visually
            different, and not overexposed. This lets a useful preference profile emerge from only a
            handful of answers.

            Group mode first normalizes each person's utilities so every traveler has equal influence, then
            averages those normalized scores. The interface also reports disagreement instead of pretending
            that a group recommendation is unanimous.
            """
        )
    with facts:
        st.markdown(
            """
            **Prototype dataset**  
            20 destinations · 10 interpretable dimensions

            **Privacy**  
            Choices stay in the current browser session. Shared profiles contain model weights, not a trip history.

            **Artwork**  
            Deterministic local SVG postcards; no image API, keys, tracking, or network connection required.

            **Browser version**<br>
            Zero-install HTML, CSS, and JavaScript; Python dependencies are only for local development.
            """
        )
    st.header("Responsible use & limitations")
    st.markdown(
        """
        This is a discovery aid, not a booking engine. The dataset is small and curated, the feature ratings are
        subjective, and destination fit is not the same as affordability, accessibility, safety, visa eligibility,
        or current local conditions.

        A natural next step is [OpenAI CLIP](https://github.com/openai/CLIP), which learns a shared representation
        for images and text. With a larger licensed photo dataset, Roam could use CLIP to recognize visual similarity
        and relate destination images to prompts such as “a quiet natural retreat” without manually rating every
        place. CLIP is deliberately not used here so this prototype stays lightweight, deterministic, explainable,
        and zero-install on the public website. A production version should also add accessible image descriptions,
        practical travel constraints, and research with travelers from different backgrounds.

        **AI-use disclosure:** Generative AI (OpenAI Codex) assisted with implementation, interface copy, tests,
        and documentation.
        """
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown("# ROAM ↗")
        st.caption("FIND YOUR SOMEWHERE")
        st.markdown("---")
        page = st.radio("Navigate", ["Discover", "My Taste", "Group Trip", "How It Works"], label_visibility="collapsed")
        st.markdown("---")
        count = len(st.session_state.comparisons)
        st.caption(f"PROFILE · {count} CHOICE{'S' if count != 1 else ''}")
        if count:
            st.progress(current_profile().confidence)
        st.markdown('<div class="footer-note">A Project 0 prototype<br>Built to make group decisions gentler.</div>', unsafe_allow_html=True)
    return page


inject_styles()
initialize_state()
selected_page = sidebar()

if selected_page == "Discover":
    discover_page()
elif selected_page == "My Taste":
    taste_page()
elif selected_page == "Group Trip":
    group_page()
else:
    about_page()
