"""A small, transparent destination dataset used by the Roam prototype.

The feature values are deliberately inspectable.  They act as compact visual-
semantic embeddings: destinations close in this space share travel qualities.
The model never uses destination names when learning a preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURES = (
    "Beach",
    "Nature",
    "Adventure",
    "Culture",
    "Food",
    "Nightlife",
    "History",
    "Relaxation",
    "Budget-friendly",
    "Cool climate",
)


@dataclass(frozen=True)
class Destination:
    slug: str
    name: str
    country: str
    region: str
    blurb: str
    best_for: str
    palette: tuple[str, str, str]
    features: tuple[float, ...]

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}"

    @property
    def image_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / "destinations" / f"{self.slug}.svg"


DESTINATIONS = (
    Destination("amalfi", "Amalfi Coast", "Italy", "Europe", "Pastel villages, cliff roads, and long lunches above the Mediterranean.", "scenery & food", ("#126E82", "#F3C98B", "#EF6F6C"), (.95,.62,.42,.76,.91,.42,.67,.86,.25,.10)),
    Destination("banff", "Banff", "Canada", "North America", "Glacial lakes and alpine trails framed by the Canadian Rockies.", "mountains & hiking", ("#16425B", "#81C3D7", "#F1FAEE"), (.05,.98,.88,.22,.35,.10,.18,.63,.45,.95)),
    Destination("kyoto", "Kyoto", "Japan", "Asia", "Quiet temples, seasonal gardens, craft traditions, and remarkable cuisine.", "tradition & calm", ("#61210F", "#E07A5F", "#F4F1DE"), (.08,.66,.28,.98,.91,.18,.98,.82,.42,.50)),
    Destination("new_orleans", "New Orleans", "USA", "North America", "Live jazz, Creole flavors, ornate streets, and joyful late nights.", "music & food", ("#4F345A", "#F4B942", "#D7263D"), (.05,.18,.22,.92,.96,.98,.82,.28,.58,.12)),
    Destination("bali", "Bali", "Indonesia", "Asia", "Surf breaks, rice terraces, warm rituals, and restorative retreats.", "wellness & beaches", ("#2A9D8F", "#E9C46A", "#F4A261"), (.94,.86,.61,.75,.72,.49,.44,.98,.75,.04)),
    Destination("reykjavik", "Reykjavík", "Iceland", "Europe", "A creative little capital beside waterfalls, volcanoes, and geothermal pools.", "wild landscapes", ("#264653", "#5BC0BE", "#E0FBFC"), (.12,.96,.84,.68,.61,.54,.40,.70,.24,.98)),
    Destination("marrakech", "Marrakech", "Morocco", "Africa", "Lantern-lit souks, courtyard gardens, desert colors, and fragrant tagines.", "markets & design", ("#A44A3F", "#D4A373", "#FAEDCD"), (.06,.38,.50,.97,.90,.35,.90,.64,.78,.08)),
    Destination("patagonia", "Patagonia", "Argentina", "South America", "Wind-carved peaks, blue ice, and vast trails at the edge of the world.", "remote adventure", ("#023047", "#8ECAE6", "#FFB703"), (.04,.99,.99,.18,.28,.05,.12,.38,.38,.92)),
    Destination("lisbon", "Lisbon", "Portugal", "Europe", "Tiled hills, Atlantic light, neighborhood cafés, and easygoing evenings.", "city breaks", ("#005F73", "#E9D8A6", "#EE9B00"), (.42,.32,.31,.89,.91,.78,.86,.72,.73,.22)),
    Destination("santorini", "Santorini", "Greece", "Europe", "Whitewashed lanes, volcanic coves, and sunsets made for slowing down.", "romance & views", ("#0077B6", "#CAF0F8", "#FFAFCC"), (.93,.52,.31,.70,.76,.48,.68,.97,.34,.08)),
    Destination("cape_town", "Cape Town", "South Africa", "Africa", "Mountains meet ocean beside vineyards, art, and a bold food scene.", "variety & outdoors", ("#006D77", "#83C5BE", "#E29578"), (.82,.91,.83,.72,.87,.70,.57,.67,.61,.28)),
    Destination("prague", "Prague", "Czechia", "Europe", "Gothic lanes, riverside walks, storied pubs, and a skyline of spires.", "architecture & value", ("#582F0E", "#936639", "#DDB892"), (.02,.25,.22,.92,.72,.83,.99,.52,.86,.70)),
    Destination("costa_rica", "Monteverde", "Costa Rica", "Central America", "Cloud forests, swinging bridges, wildlife, and waterfall adventures.", "eco-adventure", ("#1B4332", "#74C69D", "#D8F3DC"), (.48,.99,.93,.42,.48,.18,.20,.74,.64,.15)),
    Destination("seoul", "Seoul", "South Korea", "Asia", "Palaces and design districts powered by street food and all-night energy.", "culture & nightlife", ("#3A0CA3", "#F72585", "#4CC9F0"), (.03,.22,.25,.94,.96,.96,.82,.36,.57,.68)),
    Destination("queenstown", "Queenstown", "New Zealand", "Oceania", "A lakeside basecamp for alpine hikes, skiing, and big thrills.", "adrenaline & scenery", ("#003049", "#669BBC", "#FDF0D5"), (.15,.97,.99,.28,.47,.47,.20,.55,.40,.83)),
    Destination("vienna", "Vienna", "Austria", "Europe", "Grand museums, coffeehouses, concert halls, and elegant public spaces.", "arts & history", ("#7F5539", "#DDB892", "#FFF1E6"), (.01,.28,.16,.99,.88,.57,.99,.77,.43,.75)),
    Destination("tulum", "Tulum", "Mexico", "North America", "Caribbean water, jungle cenotes, Maya ruins, and barefoot evenings.", "beach & discovery", ("#007F5F", "#80B918", "#F2E8CF"), (.99,.83,.69,.66,.75,.64,.76,.88,.57,.02)),
    Destination("hanoi", "Hanoi", "Vietnam", "Asia", "A layered old quarter of tiny stools, lakes, temples, and legendary dishes.", "street food & culture", ("#9B2226", "#CA6702", "#E9D8A6"), (.05,.31,.33,.96,.99,.70,.91,.48,.96,.16)),
    Destination("swiss_alps", "Swiss Alps", "Switzerland", "Europe", "Storybook railways, pristine valleys, and high-mountain walks.", "comfort & mountains", ("#1D3557", "#A8DADC", "#F1FAEE"), (.03,.99,.79,.37,.66,.12,.33,.86,.12,.96)),
    Destination("cartagena", "Cartagena", "Colombia", "South America", "Colorful balconies, Caribbean rhythms, and golden-hour plazas.", "color & coastal energy", ("#F94144", "#F9C74F", "#43AA8B"), (.88,.38,.38,.89,.85,.88,.90,.65,.72,.03)),
)

DESTINATION_BY_SLUG = {destination.slug: destination for destination in DESTINATIONS}


def feature_matrix() -> np.ndarray:
    """Return a normalized matrix suitable for the linear preference model."""
    matrix = np.asarray([destination.features for destination in DESTINATIONS], dtype=float)
    # Centering makes a positive weight mean "more than the average destination".
    return matrix - matrix.mean(axis=0, keepdims=True)
