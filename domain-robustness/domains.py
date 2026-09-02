"""The two fixed Env 5 robustness-domain views."""

DOMAINS = {
    "hotel-cleaning": {
        "negotiation_names": ("hotel owner", "cleaning provider"),
        "negotiation_titles": ("Hotel Owner", "Cleaning Provider"),
        "negotiation_slugs": ("hotel_owner", "cleaning_provider"),
        "performance_roles": ("hotel-owner", "cleaning-provider"),
        "performance_slugs": ("hotel_owner", "cleaning_provider"),
        "counterparts": ("cleaning_provider", "hotel_owner"),
        "agent_names": ("hotel owner", "cleaning provider"),
        "money_scale": 100,
        "services": {
            "A": "Conference Room Cleaning",
            "B": "King-Sized Room Cleaning",
            "C": "Single Room Cleaning",
        },
        "materials": {
            "X": "Detergent",
            "Y": "Disinfectant",
            "Z": "Wet-wipes",
        },
    },
    "ai-model-hosting": {
        "negotiation_names": ("AI startup founder", "AI service provider"),
        "negotiation_titles": ("AI Startup Founder", "AI Service Provider"),
        "negotiation_slugs": ("ai_startup_founder", "ai_service_provider"),
        "performance_roles": ("ai-startup-founder", "ai-service-provider"),
        "performance_slugs": ("ai_startup_founder", "ai_service_provider"),
        "counterparts": ("ai_service_provider", "ai_startup_founder"),
        "agent_names": ("AI startup founder", "AI service provider"),
        "money_scale": 1_000,
        "services": {"A": "Fine-tuning", "B": "Model-hosting", "C": "Inference"},
        "materials": {"X": "CPU", "Y": "Memory", "Z": "GPU"},
    },
}

DOMAIN_NAMES = tuple(DOMAINS)
