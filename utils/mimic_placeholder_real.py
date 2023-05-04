import re
from typing import List
import names
import random
import string
import calendar
import datetime

DEID_TO_TAG = {
    "First Name": "[FIRST NAME]",
    "Last Name": "[LAST NAME]",
    "Clip Number (Radiology)": "[Reg#]",
    "Hospital": "[HOSPITAL NAME]",
    "Numeric Identifier": "[Reg#]",
    "Location": "[LOC]",
    "Initials": "[FIRST NAME]",
    "Known lastname": "[LAST NAME]",
    "Known firstname": "[FIRST NAME]",
    "Age over 90": "[Age]",
    "Medical Record Number": "[Reg#]",
    "Telephone/Fax": "[TELEPHONE]",
    "Name": "[FIRST NAME]",
    "Street Address": "[LOC]",
    "Date Range": "[DR]",
    "Date range": "[DR]",
    "Serial Number": "[Reg#]",
    "Social Security Number": "[Reg#]",
    "State": "[State]",
    "E-mail address": "[EMAIL]",
    "Company": "[COMPANY NAME]",
    "Pager number": "[TELEPHONE]",
    "Country": "[Country]",
    "MD Number": "[Reg#]",
    "Wardname": "[LOC]",
    "University/College": "[COLLEGE]",
    "Apartment Address": "[LOC]",
    "URL": "[URL]",
    "CC Contact Info": "[TELEPHONE]",
    "Attending Info": "[TELEPHONE]",
    "Job Number": "[Reg#]",
    "Year": "[YR]",
    "Month": "[MO]",
    "Day": "[DAY]",
    "Holiday": "[DAY]",
    "Unit Number": "[Reg#]",
    "PO Box": "[LOC]",
    "Dictator Info": "[FIRST NAME]",
    "Provider Number": "[Reg#]",
    "Doctor First Name": "[FIRST NAME]",
    "Doctor Last Name": "[FIRST NAME]",
    "YYYY-MM-DD": "[DATE]",
}


class GenerateReplacement:
    def __init__(self):
        self.pattern_type_function_mapping = {
            "[FIRST NAME]": self.generate_first_name,
            "[LAST NAME]": self.generate_last_name,
            "[Reg#]": self.generate_registration_number,
            "[HOSPITAL NAME]": self.generate_hospital_name,
            "[LOC]": self.generate_location,
            "[Age]": self.generate_age,
            "[TELEPHONE]": self.generate_phone_number,
            "[DR]": self.generate_date_range,
            "[State]": self.generate_state_name,
            "[EMAIL]": self.generate_email,
            "[COMPANY NAME]": self.generate_company_name,
            "[Country]": self.generate_random_country,
            "[COLLEGE]": self.generate_random_college,
            "[URL]": self.generate_random_url,
            "[YR]": self.generate_random_year,
            "[MO]": self.generate_random_month,
            "[DAY]": self.generate_random_day,
            "[DATE]": self.generate_random_date,
        }

    def generate_random_date(self):
        year = random.randint(1900, datetime.date.today().year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = datetime.date(year, month, day)
        return date.strftime("%Y-%m-%d")

    def generate_date_range(
        self,
        start_date=datetime.datetime(2022, 5, 1, 0, 0, 0),
        end_date=datetime.datetime(2052, 5, 10, 23, 59, 59),
    ):
        start_date = start_date.strftime("%B %d, %Y")
        end_date = end_date.strftime("%B %d, %Y")
        return f"{start_date} - {end_date}"

    def generate_random_day(self, month=8, year=2022):
        _, num_days = calendar.monthrange(year, month)
        return random.randint(1, num_days)

    def generate_random_year(self, start_year=1900, end_year=2100):
        return random.randint(start_year, end_year)

    def generate_random_month(self):
        month = random.randint(1, 12)
        month_name = calendar.month_name[month]
        return month_name

    def generate_first_name(self):
        return names.get_first_name()

    def generate_last_name(self):
        return names.get_last_name()

    def generate_full_name(self):
        return names.get_full_name()

    def generate_registration_number(self):
        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=6)
        )
        return random_string

    def generate_hospital_name(self):
        """
        Generates a random hospital name.
        """
        # List of possible hospital name components
        prefixes = ["General", "Community", "Regional", "City", "County", "University"]
        suffixes = [
            "Hospital",
            "Medical Center",
            "Clinic",
            "Healthcare",
            "Care Center",
            "Center",
        ]

        # Randomly select a prefix and suffix
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        # Generate a random number to append to the end of the name
        number = random.randint(1, 999)

        # Concatenate the components to create the hospital name
        hospital_name = f"{prefix} {suffix} {number}"

        return hospital_name

    def generate_location(self):
        """
        Generates a random location.
        """
        # List of possible location components
        cities = [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
        ]
        states = [
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
        ]

        # Randomly select a city and state
        city = random.choice(cities)
        state = random.choice(states)

        # Concatenate the city and state to create the location
        location = f"{city}, {state}"

        return location

    def generate_age(self):
        return random.choice(range(0, 100))

    def generate_phone_number(self, fax=False):
        """
        Generates a random US telephone or fax number.

        Args:
        - fax: a boolean value indicating whether to generate a fax number (default: False)

        Returns:
        - a string representing the telephone or fax number
        """
        # Choose the appropriate prefix for a telephone or fax number
        if fax:
            prefix = random.choice(["800", "888", "877", "866", "855", "844"])
        else:
            prefix = random.choice(["800", "888", "877", "866", "855", "844", "900"])

        # Generate random digits for the rest of the number
        digits = "".join([str(random.randint(0, 9)) for _ in range(7)])

        # Format the number with dashes
        if fax:
            number = f"{prefix}-{digits}"
        else:
            number = f"{prefix}-555-{digits}"

        return number

    def generate_state_name(
        self,
    ):
        states = [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ]
        return random.choice(states)

    def generate_email(
        self,
    ):
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        letters = string.ascii_lowercase
        username_length = random.randint(5, 10)
        username = "".join(random.choice(letters) for i in range(username_length))
        domain = random.choice(domains)
        email = f"{username}@{domain}"
        return email

    def generate_company_name(
        self,
    ):
        adjectives = [
            "Creative",
            "Dynamic",
            "Innovative",
            "Energetic",
            "Agile",
            "Collaborative",
            "Tech",
            "Strategic",
            "Global",
            "Flexible",
        ]
        nouns = [
            "Solutions",
            "Ventures",
            "Industries",
            "Enterprises",
            "Labs",
            "Partners",
            "Corp",
            "Innovations",
            "Works",
            "Group",
        ]
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        return f"{adjective} {noun}"

    def generate_random_country(
        self,
    ):
        countries = [
            "Afghanistan",
            "Albania",
            "Algeria",
            "Angola",
            "Argentina",
            "Australia",
            "Austria",
            "Bangladesh",
            "Belgium",
            "Brazil",
            "Canada",
            "Chile",
            "China",
            "Colombia",
            "Croatia",
            "Czech Republic",
            "Denmark",
            "Egypt",
            "Ethiopia",
            "Finland",
            "France",
            "Germany",
            "Ghana",
            "Greece",
            "India",
            "Indonesia",
            "Iran",
            "Iraq",
            "Ireland",
            "Israel",
            "Italy",
            "Japan",
            "Kenya",
            "Mexico",
            "Morocco",
            "Netherlands",
            "New Zealand",
            "Nigeria",
            "Norway",
            "Pakistan",
            "Philippines",
            "Poland",
            "Portugal",
            "Russia",
            "Saudi Arabia",
            "South Africa",
            "South Korea",
            "Spain",
            "Sweden",
            "Switzerland",
            "Thailand",
            "Turkey",
            "United Kingdom",
            "United States",
        ]
        return random.choice(countries)

    def generate_random_college(
        self,
    ):
        prefixes = [
            "Central",
            "North",
            "South",
            "East",
            "West",
            "National",
            "State",
            "Technical",
            "Community",
        ]
        suffixes = ["College", "University", "Institute", "Academy", "Polytechnic"]
        names = [
            "Abacus",
            "Birch",
            "Cedar",
            "Delta",
            "Echo",
            "Falcon",
            "Gamma",
            "Horizon",
            "Ivory",
            "Jupiter",
            "Kappa",
            "Lion",
            "Meadow",
            "Nova",
            "Orbit",
            "Pinnacle",
            "Quest",
            "Ridge",
            "Sierra",
            "Titan",
            "Union",
            "Vanguard",
            "Windsor",
            "Xavier",
            "Yellowstone",
            "Zenith",
        ]

        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        name = random.choice(names)

        return f"{prefix} {name} {suffix}"

    def generate_random_url(
        self,
    ):
        letters = string.ascii_lowercase
        url = "https://www."

        for i in range(10):
            url += random.choice(letters)

        url += ".com"

        return url


def is_pattern_match(pattern, text):
    """
    Check if the given string matches the pattern or no.
    If it matches the pattern then return true
    """
    match_ = re.match(pattern, text, flags=re.IGNORECASE)
    return True if match_ else False


def is_date_pattern(text):
    """
    Checks if the given string `text` matches the pattern "yyyy-m-d",
    where "yyyy" is the year, "m" is the month (1-12), and "d" is the day of the month (1-31).
    Returns True if the pattern is found, False otherwise.
    """
    pattern = r"^\d{4}-\d{1,2}-\d{1,2}$"
    match = re.match(pattern, text)
    return match is not None


def get_type_placeholders(placeholders: List[str]):
    type_placeholders = []

    for placeholder in placeholders:
        for pattern, pattern_type in DEID_TO_TAG.items():
            if is_pattern_match(pattern, placeholder):
                type_placeholders.append(pattern_type)
                break
        else:
            if is_date_pattern(placeholder):
                type_placeholders.append("[DATE]")
            else:
                print(f"There is no pattern match for the placeholder: {placeholder}")
                type_placeholders.append("[NONE]")

    return type_placeholders


def extract_placeholders(text) -> List:
    """
    Extracts placeholders of the format [*PLACEHOLDER*] from a given text string.
    Returns a list of all placeholders found in the text.
    """
    placeholders = re.findall(r"\[\*\*([^\[\]]+)\*\*\]", text)
    return placeholders


def replace_placeholder(text, placeholder, replacement):
    """
    Replaces all occurrences of the placeholder string `[**PLACEHOLDER**]` in the
    given text with the replacement string.
    """
    return text.replace(f"[**{placeholder}**]", replacement)


def remove_placeholders(note: str):
    """Given a note, remove the placeholders and replace them with real ones"""

    # Extract the placeholders from the note
    placeholders = extract_placeholders(note)
    placeholder_types = get_type_placeholders(placeholders)
    placeholder_replacements = []
    generator = GenerateReplacement()

    print(placeholders)
    print(placeholder_types)

    assert len(placeholders) == len(placeholder_types)

    for placeholder, placeholder_type in zip(placeholders, placeholder_types):
        function_ = generator.pattern_type_function_mapping.get(placeholder_type)
        if function_:
            replacement = function_()
        else:
            replacement = placeholder

        placeholder_replacements.append(replacement)

    for placeholder, placeholder_replacement in zip(
        placeholders, placeholder_replacements
    ):
        note = replace_placeholder(note, str(placeholder), str(placeholder_replacement))

    return note


if __name__ == "__main__":
    from handystuff.loaders import load_jsonl, write_jsonl
    import os
    from tqdm import tqdm

    JSONL_FILE = f"{os.environ['PROJECT_ROOT']}/unlabelled.jsonl"
    data = load_jsonl(JSONL_FILE)
    data = data[:500]

    notes = [note["section_text"].strip() for note in tqdm(data, desc="Reading notes")]

    cleaned_notes = []
    for note in tqdm(notes, desc="Removing placeholders"):
        note = note.strip()
        cleaned_note = remove_placeholders(note)
        cleaned_notes.append(cleaned_note)

    for note, cleaned_note in zip(data, cleaned_notes):
        note["cleaned_note"] = cleaned_note

    write_jsonl(data, f"{os.environ['PROJECT_ROOT']}/unlabelled_clean.jsonl")
