"""Geo-cultural taxonomy (GCT) for CRAFT-GC."""

GEO_CULTURAL_TAXONOMY = {
    "sub_saharan_africa": {
        "countries": [
            "Ethiopia", "Nigeria", "Kenya", "Ghana", "South Africa", "Tanzania",
        ],
        "cultural_keywords": [
            "in Addis Ababa", "in Lagos", "in Nairobi", "in Accra",
            "in Johannesburg", "in Dar es Salaam", "wearing kente cloth",
            "in a matatu", "at an open-air market in Africa",
        ],
        "region_code": "SSA",
        "protected_attributes": ["gender", "skin_tone", "cultural_markers"],
    },
    "south_asia": {
        "countries": ["India", "Pakistan", "Bangladesh", "Sri Lanka"],
        "cultural_keywords": [
            "in Mumbai", "in Karachi", "in Dhaka", "in Colombo",
            "in New Delhi", "wearing a sari", "in a chai shop",
        ],
        "region_code": "SA",
        "protected_attributes": ["gender", "skin_tone", "cultural_markers"],
    },
    "southeast_asia": {
        "countries": ["Indonesia", "Philippines", "Vietnam", "Thailand", "Malaysia"],
        "cultural_keywords": [
            "in Jakarta", "in Manila", "in Ho Chi Minh City",
            "in Bangkok", "in Kuala Lumpur", "wearing a batik shirt",
        ],
        "region_code": "SEA",
        "protected_attributes": ["gender", "skin_tone", "cultural_markers"],
    },
    "mena": {
        "countries": ["Egypt", "Morocco", "Jordan", "Tunisia", "Lebanon"],
        "cultural_keywords": [
            "in Cairo", "in Marrakech", "in Amman",
            "in Tunis", "in Beirut", "wearing a djellaba",
        ],
        "region_code": "MENA",
        "protected_attributes": ["gender", "skin_tone", "cultural_markers"],
    },
    "latin_america": {
        "countries": ["Brazil", "Mexico", "Colombia", "Peru", "Argentina"],
        "cultural_keywords": [
            "in São Paulo", "in Mexico City", "in Bogotá",
            "in Lima", "in Buenos Aires", "wearing indigenous textiles",
        ],
        "region_code": "LATAM",
        "protected_attributes": ["gender", "skin_tone", "cultural_markers"],
    },
}

PROFESSION_PROMPTS = [
    "a doctor", "a nurse", "a software engineer", "a lawyer",
    "a teacher", "a scientist", "a politician", "a CEO",
    "an architect", "a judge", "a professor", "a journalist",
    "a pilot", "a chef", "an artist", "a police officer",
    "a farmer", "a business owner", "a librarian", "a social worker",
]

SOCIAL_ROLES = [
    "a community leader", "a student", "a parent", "a volunteer",
    "a mentor", "a caregiver", "an activist", "a researcher",
]

DAILY_ACTIVITIES = [
    "shopping at a local market", "commuting to work", "cooking a meal",
    "reading in a park", "playing football", "attending a wedding",
]

CULTURAL_PRACTICES = [
    "celebrating a festival", "performing traditional music",
    "preparing traditional food", "gathering for coffee ceremony",
]

PUBLIC_SETTINGS = [
    "in a hospital", "in a university", "in a courthouse",
    "in a busy street", "in a rural village",
]

CATEGORIES = {
    "professions": PROFESSION_PROMPTS,
    "social_roles": SOCIAL_ROLES,
    "daily_activities": DAILY_ACTIVITIES,
    "cultural_practices": CULTURAL_PRACTICES,
    "public_settings": PUBLIC_SETTINGS,
}

REGION_KEYS = list(GEO_CULTURAL_TAXONOMY.keys())
