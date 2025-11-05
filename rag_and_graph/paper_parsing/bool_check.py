import re

def has_citation_markers(text):
    """
    Check if text contains academic citation markers in specific formats,
    excluding years and author-year citations.
    """
    # Look for citation patterns but exclude:
    # - Years (4-digit numbers starting with 19 or 20)
    # - Citations containing text (like author names)
    
    # Pattern for numeric citations (single or multiple)
    numeric_citation_pattern = r'(?:\[|\()\d{1,3}(?:,\s*\d{1,3})*(?:\)|\])'
    
    # Find all potential matches
    matches = re.findall(numeric_citation_pattern, text)
    
    # Filter out false positives
    for match in matches:
        # Extract just the numbers from the match
        numbers = re.findall(r'\d+', match)
        
        # Check if this looks like a real citation (not a year or other numeric reference)
        if is_valid_citation(numbers):
            return True
    
    return False

def is_valid_citation(numbers):
    """
    Determine if a set of numbers represents valid academic citations
    (not years, page numbers, etc.)
    """
    for num in numbers:
        num_int = int(num)
        # Exclude years (typically 4-digit, especially 19xx or 20xx)
        if len(num) == 4 and (num.startswith('19') or num.startswith('20')):
            return False
        # Exclude very large numbers that might be other references
        if num_int > 1000:
            return False
        # Exclude single-digit numbers that might be footnotes or other markers
        if len(num) == 1 and num_int < 10:
            return False
    
    return True

def check_json_for_markers(json_data):
    """Check if the JSON data contains citation markers in the text field"""
    text = json_data.get('text', '')
    return has_citation_markers(text)
