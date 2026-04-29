"""
Two-tier merchant decoder:
  1. Rules-based lookup (fast, free, covers ~80% of AU transactions)
  2. Claude fallback for cache misses (batched to minimise API calls)
"""

import re
from functools import lru_cache

from enrichment.claude_client import ClaudeClient

# Common AU bank transaction prefixes to strip before lookup
_STRIP_PREFIXES = re.compile(
    r"^(eftpos\s+|visa\s+(purchase\s+|debit\s+)?|mastercard\s+(purchase\s+)?|"
    r"sq\s*\*\s*|paypal\s*\*\s*|amex\s+|direct\s+debit\s+|dd\s+|"
    r"direct\s+credit\s+|dc\s+|bpay\s+|osko\s+|payid\s+)",
    re.IGNORECASE,
)
_TRAILING_JUNK = re.compile(r"\s+(au|nsw|vic|qld|wa|sa|act|tas|nt)\s*$", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s{2,}")

# Rules: lowercase keyword → clean name
_RULES: dict[str, str] = {
    "woolworths": "Woolworths",
    "woolies": "Woolworths",
    "coles": "Coles",
    "aldi": "ALDI",
    "iga": "IGA",
    "harris farm": "Harris Farm Markets",
    "foodworks": "Foodworks",
    "7-eleven": "7-Eleven",
    "mcdonald": "McDonald's",
    "mcdonalds": "McDonald's",
    "kfc": "KFC",
    "hungry jack": "Hungry Jack's",
    "subway": "Subway",
    "domino": "Domino's",
    "pizza hut": "Pizza Hut",
    "guzman": "Guzman y Gomez",
    "nandos": "Nando's",
    "starbucks": "Starbucks",
    "uber eats": "Uber Eats",
    "doordash": "DoorDash",
    "menulog": "Menulog",
    "uber": "Uber",
    "ola": "Ola",
    "didi": "DiDi",
    "opal": "Opal Card (Transport NSW)",
    "myki": "Myki (Public Transport Victoria)",
    "go card": "Go Card (Translink QLD)",
    "qantas": "Qantas Airways",
    "jetstar": "Jetstar Airways",
    "virgin australia": "Virgin Australia",
    "rex airline": "Rex Airlines",
    "airbnb": "Airbnb",
    "booking.com": "Booking.com",
    "expedia": "Expedia",
    "agoda": "Agoda",
    "netflix": "Netflix",
    "spotify": "Spotify",
    "stan": "Stan",
    "disney": "Disney+",
    "amazon prime": "Amazon Prime",
    "apple": "Apple",
    "google": "Google",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "ebay": "eBay",
    "kmart": "Kmart",
    "target": "Target",
    "big w": "Big W",
    "myer": "Myer",
    "david jones": "David Jones",
    "jb hi-fi": "JB Hi-Fi",
    "harvey norman": "Harvey Norman",
    "officeworks": "Officeworks",
    "bunnings": "Bunnings Warehouse",
    "ikea": "IKEA",
    "telstra": "Telstra",
    "optus": "Optus",
    "vodafone": "Vodafone",
    "tpg": "TPG",
    "aussie broadband": "Aussie Broadband",
    "agl": "AGL Energy",
    "origin energy": "Origin Energy",
    "energy australia": "EnergyAustralia",
    "sydney water": "Sydney Water",
    "icon water": "Icon Water",
    "yarra valley water": "Yarra Valley Water",
    "medibank": "Medibank Private",
    "bupa": "Bupa",
    "hcf": "HCF",
    "nib": "nib Health Funds",
    "chemist warehouse": "Chemist Warehouse",
    "priceline": "Priceline Pharmacy",
    "medicare": "Medicare",
    "ato": "Australian Taxation Office",
    "centrelink": "Services Australia (Centrelink)",
    "service nsw": "Service NSW",
    "australia post": "Australia Post",
    "auspost": "Australia Post",
    "toll": "Toll (road charge)",
    "linkt": "Linkt (tolls)",
    "e-toll": "E-Toll NSW",
    "parking": "Parking",
    "wilson parking": "Wilson Parking",
    "secure parking": "Secure Parking",
    "salary": "Salary / Wages",
    "payroll": "Payroll",
    "wages": "Wages",
    "interest": "Interest charge",
    "fee": "Bank fee",
    "overdrawn": "Overdraft fee",
    "transfer": "Bank transfer",
    "bpay": "BPAY payment",
}


def _normalise(text: str) -> str:
    text = _STRIP_PREFIXES.sub("", text)
    text = _TRAILING_JUNK.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    return text


@lru_cache(maxsize=1000)
def _rules_lookup(normalised: str) -> str | None:
    for keyword, clean_name in _RULES.items():
        if keyword in normalised:
            return clean_name
    return None


class MerchantDecoder:
    def __init__(self, use_claude: bool = True) -> None:
        self._claude = ClaudeClient() if use_claude else None
        self._cache: dict[str, str] = {}

    def decode_batch(self, descriptions: list[str]) -> dict[str, str]:
        """Return {raw_description: clean_merchant_name} for every input."""
        result: dict[str, str] = {}
        claude_needed: list[str] = []

        for desc in descriptions:
            if desc in self._cache:
                result[desc] = self._cache[desc]
                continue

            norm = _normalise(desc)
            lookup = _rules_lookup(norm)
            if lookup:
                result[desc] = lookup
                self._cache[desc] = lookup
            else:
                claude_needed.append(desc)

        if claude_needed and self._claude:
            claude_result = self._claude.decode_merchants(claude_needed)
            for raw, clean in claude_result.items():
                result[raw] = clean
                self._cache[raw] = clean

        # Fallback: use normalised description for anything still unresolved
        for desc in descriptions:
            if desc not in result:
                norm = _normalise(desc).title()
                result[desc] = norm or "Unknown Merchant"
                self._cache[desc] = result[desc]

        return result
