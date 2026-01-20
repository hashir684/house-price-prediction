import re
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="tinyllama", temperature=0)

def extract_details_from_query(user_query):

    prompt = f"""
You are a helper that reads real estate queries.

User query: "{user_query}"

From this sentence, identify:

- location
- square feet
- number of bathrooms
- number of BHK

Reply in ONE simple line like this:

Location: <location>, Sqft: <number>, Bath: <number>, BHK: <number>

Do not write anything else.
"""

    response = llm.invoke(prompt)

    text = response.lower()

    # Extract numeric values using regex
    sqft = re.search(r"sqft[:\s]+(\d+)", text)
    bath = re.search(r"bath[:\s]+(\d+)", text)
    bhk = re.search(r"bhk[:\s]+(\d+)", text)

    location = None
    loc_match = re.search(r"location[:\s]+([a-zA-Z\s]+),", text)

    if loc_match:
        location = loc_match.group(1).strip().title()

    result = {
        "location": location,
        "total_sqft": int(sqft.group(1)) if sqft else None,
        "bath": int(bath.group(1)) if bath else None,
        "bhk": int(bhk.group(1)) if bhk else None
    }

    return result
