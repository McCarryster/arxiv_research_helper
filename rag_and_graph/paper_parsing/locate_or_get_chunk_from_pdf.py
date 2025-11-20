from typing import List, Tuple
import re
from tqdm import tqdm
from PyPDF2 import PdfReader


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize `text` into word tokens and return list of tuples (word, start_char_index, end_char_index).
    Uses regex to capture words and common contractions. Indices are in the original `text`.
    """
    # \w includes underscore; allow apostrophes in contractions
    pattern = re.compile(r"\w+(?:['’]\w+)*", flags=re.UNICODE)
    tokens: List[Tuple[str, int, int]] = []
    for m in pattern.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def _find_sequence_token_index(pdf_tokens: List[Tuple[str, int, int]],
                               target_words: List[str]) -> List[int]:
    """
    Return list of token indexes in pdf_tokens where the sequence target_words starts.
    Matches case-insensitively on the token text (normalized by lower()).
    """
    n = len(pdf_tokens)
    p = len(target_words)
    if p == 0:
        return []

    # lower-case target words for matching
    target_lower = [w.lower() for w in target_words]

    matches: List[int] = []
    # Use tqdm to iterate token indexes (helpful for large PDFs)
    for i in tqdm(range(0, n - p + 1), desc="searching tokens", unit="i"):
        # quick check first token match, then slice compare
        if pdf_tokens[i][0].lower() != target_lower[0]:
            continue
        ok = True
        for k in range(1, p):
            if pdf_tokens[i + k][0].lower() != target_lower[k]:
                ok = False
                break
        if ok:
            matches.append(i)
    return matches

# To locate
def locate_chunk_in_pdf(chunk: str, pdf_path: str, n_first: int, n_last: int) -> Tuple[int, int]:
    """
    Locate the chunk in the PDF using its first `n_first` words and last `n_last` words.

    Returns:
        (start_offset, end_offset) -- both ints referring to character offsets within the text
        extracted from the PDF (start inclusive, end exclusive).
        If not found, returns (-1, -1).

    Notes:
    - Offsets correspond to the PDF's extracted text obtained by PdfReader (pages joined with '\n\n').
    - The function attempts robust token-level matching (case-insensitive).
    - If chunk has fewer words than requested, it will use whatever is available for first/last.
    """
    if n_first < 0 or n_last < 0:
        raise ValueError("n_first and n_last must be non-negative integers")

    # --- extract text from PDF ---
    reader = PdfReader(pdf_path)
    pages_text: List[str] = []
    for page in reader.pages:
        # PyPDF2's extract_text() may return None for empty pages; coerce to empty string
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    # join pages with explicit separators so offsets map to the joined text
    pdf_text = "\n\n".join(pages_text)

    # fix common hyphenation introduced by PDF line breaks: join "-\n" into nothing
    pdf_text = pdf_text.replace("-\n", "")

    # --- tokenize chunk to obtain first/last words ---
    chunk_tokens = tokenize_with_spans(chunk)
    chunk_words = [tok for (tok, s, e) in chunk_tokens]

    if len(chunk_words) == 0:
        return (-1, -1)

    # adjust n_first/n_last if chunk shorter
    if n_first > len(chunk_words):
        n_first_eff = len(chunk_words)
    else:
        n_first_eff = n_first

    if n_last > len(chunk_words):
        n_last_eff = len(chunk_words)
    else:
        n_last_eff = n_last

    first_seq = chunk_words[:n_first_eff]
    last_seq = chunk_words[-n_last_eff:] if n_last_eff > 0 else []

    # --- tokenize pdf text with spans ---
    pdf_tokens = tokenize_with_spans(pdf_text)
    if len(pdf_tokens) == 0:
        return (-1, -1)

    # --- find candidate start token indices for first_seq and last_seq ---
    first_matches = _find_sequence_token_index(pdf_tokens, first_seq)
    if not first_matches:
        # no match for first sequence
        return (-1, -1)

    if len(last_seq) == 0:
        # If last_seq is empty, return start of first match and end of chunk-length tokens window if possible
        # Here we simply return the span of the first sequence found
        fi = first_matches[0]
        start_char = pdf_tokens[fi][1]
        end_char = pdf_tokens[fi + len(first_seq) - 1][2]
        return (start_char, end_char)

    last_matches_all = _find_sequence_token_index(pdf_tokens, last_seq)
    if not last_matches_all:
        # no match for last sequence
        return (-1, -1)

    # --- choose a pair (first_match_index, last_match_index) where last occurs at/after first ---
    chosen_pair: Tuple[int, int] = (-1, -1)
    for fi in first_matches:
        # find the earliest last_match that has token index >= fi (it should ideally be after the first sequence)
        for lj in last_matches_all:
            # Ensure last match comes at or after first match start (we allow equal only if sequences overlap properly)
            if lj >= fi:
                chosen_pair = (fi, lj)
                break
        if chosen_pair != (-1, -1):
            break

    if chosen_pair == (-1, -1):
        return (-1, -1)

    fi, lj = chosen_pair
    start_char = pdf_tokens[fi][1]
    end_token_index = lj + len(last_seq) - 1
    if end_token_index >= len(pdf_tokens):
        return (-1, -1)
    end_char = pdf_tokens[end_token_index][2]

    return (start_char, end_char)

# To get using location start and end
def get_text_by_offsets(pdf_path: str, start_offset: int, end_offset: int) -> str:
    """
    Extract and return the substring of text from a PDF corresponding to character offsets
    in the PDF's extracted text. Offsets are interpreted the same way as in
    `locate_chunk_in_pdf()` (i.e. pages joined with "\n\n" and common hyphenation fixed).

    Args:
        pdf_path: Path to the PDF file.
        start_offset: Integer start character offset (inclusive).
        end_offset: Integer end character offset (exclusive).

    Returns:
        The extracted substring from the PDF text.

    Raises:
        ValueError: If offsets are invalid or PDF has no extractable text.
        FileNotFoundError / OSError: If the PDF path cannot be opened by PdfReader.
    """
    if not isinstance(start_offset, int) or not isinstance(end_offset, int):
        raise ValueError("start_offset and end_offset must be integers")

    if start_offset < 0 or end_offset < 0:
        raise ValueError("start_offset and end_offset must be non-negative")

    if end_offset <= start_offset:
        raise ValueError("end_offset must be greater than start_offset")

    # --- extract text from PDF pages ---
    reader = PdfReader(pdf_path)
    pages_text = []
    # Use tqdm in case the PDF is large (progress feedback for long file iterating)
    for page in tqdm(reader.pages, desc="extracting pages", unit="page"):
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    if not pages_text:
        raise ValueError("No extractable text found in PDF")

    # Build the single text blob in the same manner as locator function
    pdf_text = "\n\n".join(pages_text)
    # Fix typical PDF hyphenation artifacts introduced by line breaks
    pdf_text = pdf_text.replace("-\n", "")

    # Validate offsets against the pdf_text length
    total_len = len(pdf_text)
    if start_offset >= total_len:
        raise ValueError(f"start_offset {start_offset} is outside the PDF text range (length {total_len})")
    if end_offset > total_len:
        raise ValueError(f"end_offset {end_offset} is outside the PDF text range (length {total_len})")

    # Return the requested slice (end exclusive)
    return pdf_text[start_offset:end_offset]


# # Example usage:
# if __name__ == "__main__":
#     # Example chunk:
#     sample_chunk = """
#         Recurrent neural networks (RNNs) are a rich class of dynamic models that have been used to generate sequences in domains as diverse as music [6,4], text [30] and motion capture data [29]. RNNs can be trained for sequence generation by processing real data sequences one step at a time and predicting what comes next. Assuming the predictions are probabilistic, novel sequences can be generated from a trained network by iteratively sampling from the network's output distribution, then feeding in the sample as input at the next step. In other words by making the network treat its inventions as if they were real, much like a person dreaming. Although the network itself is deterministic, the stochasticity injected by picking samples induces a distribution over sequences. This distribution is conditional, since the internal state of the network, and hence its predictive distribution, depends on the previous inputs.

#         RNNs are 'fuzzy' in the sense that they do not use exact templates from the training data to make predictions, but rather-like other neural networksuse their internal representation to perform a high-dimensional interpolation between training examples. This distinguishes them from n-gram models and compression algorithms such as Prediction by Partial Matching [5], whose predictive distributions are determined by counting exact matches between the recent history and the training set. The result-which is immediately appar-ent from the samples in this paper-is that RNNs (unlike template-based algorithms) synthesise and reconstitute the training data in a complex way, and rarely generate the same thing twice. Furthermore, fuzzy predictions do not suffer from the curse of dimensionality, and are therefore much better at modelling real-valued or multivariate data than exact matches.

#         In principle a large enough RNN should be sufficient to generate sequences of arbitrary complexity. In practice however, standard RNNs are unable to store information about past inputs for very long [15]. As well as diminishing their ability to model long-range structure, this 'amnesia' makes them prone to instability when generating sequences. The problem (common to all conditional generative models) is that if the network's predictions are only based on the last few inputs, and these inputs were themselves predicted by the network, it has little opportunity to recover from past mistakes. Having a longer memory has a stabilising effect, because even if the network cannot make sense of its recent history, it can look further back in the past to formulate its predictions. The problem of instability is especially acute with real-valued data, where it is easy for the predictions to stray from the manifold on which the training data lies. One remedy that has been proposed for conditional models is to inject noise into the predictions before feeding them back into the model [31], thereby increasing the model's robustness to surprising inputs. However we believe that a better memory is a more profound and effective solution.

#         Long Short-term Memory (LSTM) [16] is an RNN architecture designed to be better at storing and accessing information than standard RNNs. LSTM has recently given state-of-the-art results in a variety of sequence processing tasks, including speech and handwriting recognition [10,12]. The main goal of this paper is to demonstrate that LSTM can use its memory to generate complex, realistic sequences containing long-range structure.

#         Section 2 defines a 'deep' RNN composed of stacked LSTM layers, and explains how it can be trained for next-step prediction and hence sequence generation. Section 3 applies the prediction network to text from the Penn Treebank and Hutter Prize Wikipedia datasets. The network's performance is competitive with state-of-the-art language models, and it works almost as well when predicting one character at a time as when predicting one word at a time. The highlight of the section is a generated sample of Wikipedia text, which showcases the network's ability to model long-range dependencies. Section 4 demonstrates how the prediction network can be applied to real-valued data through the use of a mixture density output layer, and provides experimental results on the IAM Online Handwriting Database. It also presents generated handwriting samples proving the network's ability to learn letters and short words direct from pen traces, and to model global features of handwriting style. Section 5 introduces an extension to the prediction network that allows it to condition its outputs on a short annotation sequence whose alignment with the predictions is unknown. This makes it suitable for handwriting synthesis, where a human user inputs a text and the algorithm generates a handwritten version of it. The synthesis network is trained on the IAM database, then used to generate cursive handwriting samples, some of which cannot be distinguished from real data by the Figure 1: Deep recurrent neural network prediction architecture. The circles represent network layers, the solid lines represent weighted connections and the dashed lines represent predictions. naked eye. A method for biasing the samples towards higher probability (and greater legibility) is described, along with a technique for 'priming' the samples on real data and thereby mimicking a particular writer's style. Finally, concluding remarks and directions for future work are given in Section 6.
#         """
#     # Replace 'example.pdf' with your PDF path
#     pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf"
#     # Get offsets for first 3 and last 3 words
#     try:
#         start_offset, end_offset = locate_chunk_in_pdf(sample_chunk, pdf_path, n_first=10, n_last=10)
#         print("start_offset:", start_offset, "end_offset:", end_offset)
#     except Exception as exc:
#         print("Error:", exc)
#     # start = 26  # example start offset
#     # end = 6194    # example end offset
#     try:
#         snippet = get_text_by_offsets(pdf_path, start_offset, end_offset)
#         print("Extracted text (length {}):\n".format(len(snippet)), snippet)
#     except Exception as e:
#         print("Error extracting text by offsets:", e)