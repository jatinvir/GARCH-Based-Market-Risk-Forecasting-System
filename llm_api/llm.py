import re


TOOLS = {
    "explain": {
        "schema": {
            "name": "explain",
            "description": "Explain a term from the project glossary (aliasesnot that  supported).",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Indicator/feature name, e.g. 'bbp20' or 'adjusted_close'."}
                },
                "required": ["term"]
            }
        }
    }
}