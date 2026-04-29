from extraction.models import Transaction, TransactionCategory, TransactionType

_RULES: list[tuple[TransactionCategory, list[str]]] = [
    (TransactionCategory.SALARY, ["salary", "payroll", "wages", "pay slip", "payslip"]),
    (TransactionCategory.GOVERNMENT, ["ato", "centrelink", "service nsw", "service vic", "medicare", "dvs", "council rates", "council"]),
    (TransactionCategory.GROCERIES, ["woolworths", "coles", "aldi", "iga", "harris farm", "foodworks", "7-eleven"]),
    (TransactionCategory.DINING, ["mcdonald", "kfc", "hungry jack", "subway", "domino", "pizza", "nandos", "guzman", "starbucks", "cafe", "restaurant", "bakery", "sushi", "thai", "chinese", "indian", "burger", "bar and grill", "bistro", "tavern", "hotel"]),
    (TransactionCategory.TRANSPORT, ["uber", "ola", "didi", "taxi", "opal", "myki", "go card", "qantas", "jetstar", "virgin australia", "rex airline", "parking", "linkt", "e-toll", "toll", "petrol", "bp ", "shell", "ampol", "caltex", "7-eleven"]),
    (TransactionCategory.UTILITIES, ["agl", "origin energy", "energy australia", "sydney water", "icon water", "yarra valley", "telstra", "optus", "vodafone", "tpg", "aussie broadband", "nbn"]),
    (TransactionCategory.HEALTH, ["chemist warehouse", "priceline", "medicare", "medibank", "bupa", "hcf", "nib", "doctor", "dental", "hospital", "pharmacy", "pathology", "radiology", "physio", "optometrist"]),
    (TransactionCategory.ENTERTAINMENT, ["netflix", "spotify", "stan", "disney", "amazon prime", "apple tv", "binge", "foxtel", "event cinema", "hoyts", "village cinema", "ticketek", "ticketmaster"]),
    (TransactionCategory.SHOPPING, ["amazon", "ebay", "kmart", "target", "big w", "myer", "david jones", "jb hi-fi", "harvey norman", "officeworks", "bunnings", "ikea", "apple store", "the iconic", "asos"]),
    (TransactionCategory.TRAVEL, ["airbnb", "booking.com", "expedia", "agoda", "hotels.com", "qantas hotel", "wotif"]),
    (TransactionCategory.FINANCE, ["interest", "bank fee", "overdrawn", "late fee", "annual fee", "insurance", "allianz", "nrma", "gio", "suncorp", "aami", "bpay"]),
    (TransactionCategory.TRANSFER, ["transfer", "payment to", "payment from", "direct credit", "direct debit", "osko", "payid"]),
]


def categorise(transaction: Transaction) -> TransactionCategory:
    text = (transaction.description_clean or transaction.description_raw).lower()

    if transaction.type == TransactionType.CREDIT:
        for keyword in ["salary", "payroll", "wages"]:
            if keyword in text:
                return TransactionCategory.SALARY

    for category, keywords in _RULES:
        if any(kw in text for kw in keywords):
            return category

    return TransactionCategory.OTHER
