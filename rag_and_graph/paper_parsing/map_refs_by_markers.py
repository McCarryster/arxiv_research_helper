import re
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Any


class ReferenceMatcher:
    """
    Improved reference extractor & matcher for arXiv-style PDFs.

    Improvements vs your original:
    - Finds the reference section by searching for common headings case-insensitively
      and using the last occurrence (references are usually at the end).
    - If headings not found, restricts to the tail of the document (avoid matching body code).
    - Supports both "[1] ..." and "1. ..." reference styles.
    - More robust marker parsing: extracts any digit sequences and ranges from inside brackets,
      so "[4,[7][8]" => 4,7,8; also supports [5,6] and ranges 1-3.
    - Avoids greedy capture across the entire document.
    """

    def __init__(self, tail_chars: int = 40000):
        """
        :param tail_chars: number of characters from the end of the PDF to consider
                           when the reference heading cannot be found. Default 40k.
        """
        # Marker detection in the text to be matched (example_json["text"])
        # We only need one general pattern; we'll extract digits/ranges from inside.
        self.marker_pattern = re.compile(r'\[([^\]]+)\]')  # capture whatever inside brackets

        # Candidate section headings (common variations)
        self.reference_section_markers = [
            "REFERENCES", "REFERENCES AND NOTES", "BIBLIOGRAPHY",
            "LITERATURE CITED", "REFERENCE LIST", "CITATIONS", "REFERENCES & NOTES"
        ]

        # How much of the end of the document to consider as "reference section" if heading not found
        self.tail_chars = tail_chars

        # Regex patterns to find reference entries inside the reference section text.
        # We look for entries starting on a new line (or start of string) to avoid grabbing body text.
        # Pattern 1: [1] Reference text
        self._bracketed_ref_pattern = re.compile(
            r'(?:^|\n)\s*\[\s*(\d+)\s*\]\s*(.*?)\s*(?=(?:\n\s*\[\s*\d+\s*\])|(?:\n\s*\d+\s*[\.\)])|$)',
            re.DOTALL
        )
        # Pattern 2: 1. Reference text
        self._number_dot_ref_pattern = re.compile(
            r'(?:^|\n)\s*(\d+)\s*\.\s*(.*?)\s*(?=(?:\n\s*\d+\s*\.)|(?:\n\s*\[\s*\d+\s*\])|$)',
            re.DOTALL
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file (concatenate pages)."""
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            # get_text() may return unicode; ensure string concatenation is safe
            text_parts.append(page.get_text())  # type: ignore
        doc.close()
        return "\n".join(text_parts)

    def _locate_reference_section(self, full_text: str) -> str:
        """
        Find the reference section text block. Strategy:
        1) Try to find last occurrence (case-insensitive) of any known reference heading.
        2) If not found, take the tail (last self.tail_chars characters) of the document as fallback.
        """
        lower_text = full_text.lower()
        found_index = -1
        found_marker = None

        for marker in self.reference_section_markers:
            idx = lower_text.rfind(marker.lower())
            if idx != -1 and idx > found_index:
                found_index = idx
                found_marker = marker

        if found_index != -1:
            # Return from the found marker to the end
            return full_text[found_index:]
        else:
            # fallback: return only the tail of the document (avoid entire doc)
            if len(full_text) <= self.tail_chars:
                return full_text
            return full_text[-self.tail_chars:]

    def extract_references_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract references from text assumed to contain the reference section (or the tail).
        Returns dict mapping reference number (str) -> reference text (cleaned).
        """
        references: Dict[str, str] = {}
        ref_section_text = self._locate_reference_section(text)

        # First try to extract bracketed references like "[1] Author..."
        for pat in (self._bracketed_ref_pattern, self._number_dot_ref_pattern):
            for match in pat.finditer(ref_section_text):
                num = match.group(1).strip()
                body = match.group(2).strip()
                # normalize whitespace
                body = re.sub(r'\s+', ' ', body).strip()
                # If we already have this reference (from previous pattern), don't overwrite
                if num not in references:
                    references[num] = body

        # If we didn't find anything with the anchored patterns, try a looser fallback:
        if not references:
            # Find bracketed numbers in the ref_section_text and grab following line(s).
            # This aims to be conservative: capture up to the next blank line or next bracketed number.
            loose_pat = re.compile(r'\[\s*(\d+)\s*\]\s*(.*?)\s*(?=(?:\n\s*\[\s*\d+\s*\])|(?:\n\s*$)|$)', re.DOTALL)
            for match in loose_pat.finditer(ref_section_text):
                num = match.group(1).strip()
                body = match.group(2).strip()
                body = re.sub(r'\s+', ' ', body).strip()
                if num not in references:
                    references[num] = body

        # Final clean-up: sometimes references include trailing page headers/footers or "arXiv..." lines;
        # do some light trimming heuristics.
        for k, v in list(references.items()):
            # remove stray "arXiv:..." fragments at the start/end of the entry
            v = re.sub(r'\s*arXiv[:\s]\S+\s*', ' ', v, flags=re.IGNORECASE)
            # strip trailing hyphenation artifacts or weird single letters at the end
            v = re.sub(r'\s{2,}', ' ', v).strip()
            references[k] = v

        return references

    def find_markers_in_text(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        Find all reference markers in the given text.
        Returns a list of tuples: (marker_text, list_of_reference_numbers)
        The marker_text is exactly what was matched (e.g. "[1]", "[5,6]", "[1][2][3]").
        Numbers can include ranges like 1-3 which are expanded into all integers in that range.
        """
        markers: List[Tuple[str, List[int]]] = []

        for match in self.marker_pattern.finditer(text):
            full_marker = match.group(0)   # e.g. "[5,6]" or "[1]"
            inner = match.group(1)         # the content inside the brackets
            # Extract digit tokens and ranges robustly from inner text:
            raw_numbers = re.findall(r'\d+(?:-\d+)?', inner)
            numbers: List[int] = []
            for token in raw_numbers:
                if '-' in token:
                    start_s, end_s = token.split('-', 1)
                    # ensure digits
                    if start_s.isdigit() and end_s.isdigit():
                        start = int(start_s)
                        end = int(end_s)
                        if end >= start:
                            numbers.extend(list(range(start, end + 1)))
                else:
                    if token.isdigit():
                        numbers.append(int(token))
            if numbers:
                markers.append((full_marker, numbers))

        return markers

    def match_references(self, pdf_path: str, example_json: Dict[str, Any], as_list: bool = False) -> Dict[str, Any]:
        """
        Main function to match references from PDF to text markers.

        Args:
            pdf_path (str): Path to the PDF file.
            example_json (Dict[str, Any]): Example JSON with "text" and "section".
            as_list (bool): If True, return references as a flat list instead of dict.
        """
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)

        # Extract references (dictionary num->text)
        references = self.extract_references_from_text(pdf_text)

        # Markers from the example text
        text_to_match = example_json.get("text", "")
        markers = self.find_markers_in_text(text_to_match)

        # Create mapping: marker_text -> list of "[n] reference text" or "Reference not found"
        mapping: Dict[str, List[str]] = {}
        for marker_text, ref_numbers in markers:
            ref_texts: List[str] = []
            for ref_num in ref_numbers:
                ref_key = str(ref_num)
                if ref_key in references:
                    ref_texts.append(f"[{ref_key}] {references[ref_key]}")
                else:
                    ref_texts.append(f"[{ref_key}] Reference not found")
            if ref_texts:
                # If the same textual marker appears multiple times, we keep the first occurrence.
                # If you prefer to collect multiple occurrences, change this to append to a list of lists.
                if marker_text not in mapping:
                    mapping[marker_text] = ref_texts

        # Format output
        if as_list:
            flat_references = [ref for refs in mapping.values() for ref in refs]
            reference_output = flat_references
        else:
            reference_output = mapping

        return {
            "section": example_json.get("section", ""),
            "text_markers": markers,
            "reference_mapping": reference_output,
            "total_references_found": len(references),
            "total_markers_matched": len(mapping)
        }


def match_refs_by_marker(pdf_path: str, section_json: Dict[str, Any], as_list: bool = False) -> Dict[str, Any]:
    matcher = ReferenceMatcher()
    return matcher.match_references(pdf_path, section_json, as_list)


# Example usage and test function
def test_solution():
    
    example_json_1 = {'section': 'Introduction', 'text': "Recurrent neural networks (RNNs) are sequence-based models of key importance for natural language understanding, language generation, video processing, and many other tasks [1][2][3]. The model's input is a sequence of symbols, where at each time step a simple neural network (RNN unit) is applied to a single symbol, as well as to the network's output from the previous time step. RNNs are powerful models, showing superb performance on many tasks, but overfit quickly. Lack of regularisation in RNN models makes it difficult to handle small data, and to avoid overfitting researchers often use early stopping, or small and under-specified models [4].\n\nDropout is a popular regularisation technique with deep networks [5,6] where network units are randomly masked during training (dropped). But the technique has never been applied successfully to RNNs. Empirical results have led many to believe that noise added to recurrent layers (connections between RNN units) will be amplified for long sequences, and drown the signal [4]. Consequently, existing research has concluded that the technique should be used with the inputs and outputs of the RNN alone [4,[7][8][9][10]. But this approach still leads to overfitting, as is shown in our experiments.\n\nRecent results at the intersection of Bayesian research and deep learning offer interpretation of common deep learning techniques through Bayesian eyes [11][12][13][14][15][16]. This Bayesian view of deep learning allowed the introduction of new techniques into the field, such as methods to obtain principled uncertainty estimates from deep learning networks [14,17]. Gal and Ghahramani [14] for example showed that dropout can be interpreted as a variational approximation to the posterior of a Bayesian neural network (NN). Their variational approximating distribution is a mixture of two Gaussians with small variances, with the mean of one Gaussian fixed at zero. This grounding of dropout in approximate Bayesian inference suggests that an extension of the theoretical results might offer insights into the use of the technique with RNN models.\n\nHere we focus on common RNN models in the field (LSTM [18], GRU [19]) and interpret these as probabilistic models, i.e. as RNNs with network weights treated as random variables, and with arXiv:1512.05287v5 [stat.ML] 5 Oct 2016 suitably defined likelihood functions. We then perform approximate variational inference in these probabilistic Bayesian models (which we will refer to as Variational RNNs). Approximating the posterior distribution over the weights with a mixture of Gaussians (with one component fixed at zero and small variances) will lead to a tractable optimisation objective. Optimising this objective is identical to performing a new variant of dropout in the respective RNNs.\n\nIn the new dropout variant, we repeat the same dropout mask at each time step for both inputs, outputs, and recurrent layers (drop the same network units at each time step). This is in contrast to the existing ad hoc techniques where different dropout masks are sampled at each time step for the inputs and outputs alone (no dropout is used with the recurrent connections since the use of different masks with these connections leads to deteriorated performance). Our method and its relation to existing techniques is depicted in figure 1. When used with discrete inputs (i.e. words) we place a distribution over the word embeddings as well. Dropout in the word-based model corresponds then to randomly dropping word types in the sentence, and might be interpreted as forcing the model not to rely on single words for its task.\n\nWe next survey related literature and background material, and then formalise our approximate inference for the Variational RNN, resulting in the dropout variant proposed above. Experimental results are presented thereafter."}
    # pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf"
    pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf" # markers
    # This would be used with an actual PDF file
    result = match_refs_by_marker(pdf_path, example_json_1)
    # print(result.items())
    for key, val in result['reference_mapping'].items():
        print(key, '=', val)
        print('-'*100)
        break

if __name__ == "__main__":
    test_solution()