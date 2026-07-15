ROLES = {
    "admin": {
        "label": "Administrator",
        "allowed_domains": ["*"],  # wildcard = unrestricted, sees everything
    },
    "software_engineer": {
        "label": "Software Engineer",
        "allowed_domains": ["software_engineering"],
    },
    
}

def get_all_domains() -> list[str]:
    """All concrete domains across every role, excluding the '*' wildcard."""
    domains = set()
    for role_info in ROLES.values():
        for d in role_info["allowed_domains"]:
            if d != "*":
                domains.add(d)
    return sorted(domains)


def get_allowed_domains(role: str) -> list[str]:
    return ROLES.get(role, {}).get("allowed_domains", [])


def is_unrestricted(role: str) -> bool:
    return "*" in get_allowed_domains(role)