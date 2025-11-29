import json
import time
import sys
import traceback
from typing import Any, Dict, Optional
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone

# === Configuration ===
BASE_URL = "https://oaipmh.arxiv.org/oai"
METADATA_PREFIX = "arXiv"
OUTFILE = "arxiv_metadata.ndjson"
CHECKPOINT = "harvest_checkpoint.json"

INITIAL_SLEEP = 0.25
MAX_RETRIES = 6
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT = 60

# defensive thresholds
MAX_CONSECUTIVE_SAME_TOKEN = 3   # if same next_token observed this many times with no progress -> fallback
MAX_CONSECUTIVE_NO_PROGRESS = 3  # if we request pages and none write new records this many times -> fallback
CHECKPOINT_EVERY = 1000          # save progress every N records written

# === XML helpers ===
def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def element_to_python(obj: ET.Element) -> Any:
    children = list(obj)
    if not children:
        text = obj.text.strip() if obj.text and obj.text.strip() else ""
        return text
    out: Dict[str, Any] = {}
    for child in children:
        key = local_name(child.tag)
        val = element_to_python(child)
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(val)
        else:
            out[key] = val
    return out

def parse_records_from_xml(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    for record in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
        header = record.find("{http://www.openarchives.org/OAI/2.0/}header")
        if header is None:
            continue
        header_info = {}
        ident = header.find("{http://www.openarchives.org/OAI/2.0/}identifier")
        datestamp = header.find("{http://www.openarchives.org/OAI/2.0/}datestamp")
        header_info["identifier"] = ident.text if ident is not None else None
        header_info["datestamp"] = datestamp.text if datestamp is not None else None
        sets = []
        for s in header.findall("{http://www.openarchives.org/OAI/2.0/}setSpec"):
            if s.text:
                sets.append(s.text)
        header_info["sets"] = sets

        metadata_elem = record.find("{http://www.openarchives.org/OAI/2.0/}metadata")
        if metadata_elem is None:
            metadata_parsed = None
        else:
            inner = list(metadata_elem)
            if inner:
                metadata_parsed = element_to_python(inner[0])
            else:
                metadata_parsed = element_to_python(metadata_elem)

        yield header_info, metadata_parsed

def get_resumption_token_from_xml(xml_bytes: bytes) -> Optional[str]:
    root = ET.fromstring(xml_bytes)
    rt = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
    if rt is None:
        return None
    text = rt.text.strip() if rt.text else ""
    return text or None

def get_resumption_token_info(xml_bytes: bytes) -> Dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    rt = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
    info = {}
    if rt is not None:
        info['text'] = rt.text.strip() if rt.text else ""
        # copy attributes like expirationDate and cursor if present
        info.update(rt.attrib)
    return info

# === Checkpoint helpers ===
def save_checkpoint(token: Optional[str],
                    token_info: Dict[str, Any],
                    out_file: str,
                    progress: Dict[str, Any]):
    d = {
        "resumptionToken": token,
        "resumptionTokenInfo": token_info,
        "outfile": out_file,
        "progress": progress,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        Path(CHECKPOINT).write_text(json.dumps(d, ensure_ascii=False, indent=2))
    except Exception as e:
        print("[WARN] Failed to write checkpoint:", e, file=sys.stderr)

def load_checkpoint() -> Dict[str, Any]:
    p = Path(CHECKPOINT)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def token_is_expired(token_info: Dict[str, Any]) -> bool:
    exp = token_info.get("expirationDate")
    if not exp:
        return False
    try:
        dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        return dt < datetime.now(timezone.utc)
    except Exception:
        return False

# === Harvest loop with improved handling of stuck tokens ===
def harvest_all(out_path: str = OUTFILE,
                start_resumption_token: Optional[str] = None,
                start_token_info: Optional[Dict[str, Any]] = None,
                last_identifier: Optional[str] = None,
                last_datestamp: Optional[str] = None):
    session = requests.Session()
    resumption_token = start_resumption_token
    token_info = start_token_info or {}
    # Load cumulative progress (if any)
    total_written_so_far = 0
    if last_identifier or last_datestamp:
        # if checkpoint provided, try to use its records_written if present
        cp = load_checkpoint()
        total_written_so_far = cp.get("progress", {}).get("records_written", 0)

    total_records_written_this_run = 0          # only for logging this process run
    total_records = total_written_so_far       # cumulative including previous runs (accurate)
    progress = {"records_written": total_records, "last_token": resumption_token,
                "last_identifier": last_identifier, "last_datestamp": last_datestamp}

    # Determine initial mode
    mode = "token" if resumption_token and not token_is_expired(token_info) else ("from_date" if last_datestamp else "token")

    if resumption_token and token_is_expired(token_info):
        if last_datestamp:
            print("[INFO] Saved resumptionToken is expired. Falling back to resume-by-date using last_datestamp.", file=sys.stderr)
            mode = "from_date"
            resumption_token = None
        else:
            print("[WARN] Saved resumptionToken is expired and no last_datestamp available. Deleting token and starting fresh.", file=sys.stderr)
            resumption_token = None
            mode = "token"

    # skip_until_identifier used both for date-resume and token-resume safety
    skip_until_identifier = last_identifier

    # variables to detect stuck behaviour
    previous_next_token = None
    consecutive_same_token = 0
    consecutive_no_progress = 0

    # We open file in append mode. We'll avoid writing duplicates by honoring skip_until_identifier.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fout:
        attempt = 0
        while True:
            try:
                if mode == "token" and resumption_token:
                    params = {"verb": "ListRecords", "resumptionToken": resumption_token}
                elif mode == "from_date" and last_datestamp:
                    params = {"verb": "ListRecords", "metadataPrefix": METADATA_PREFIX, "from": last_datestamp}
                else:
                    params = {"verb": "ListRecords", "metadataPrefix": METADATA_PREFIX}

                resp = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 503:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_for = int(retry_after) if retry_after and retry_after.isdigit() else 10
                    print(f"[WARN] 503 received. Sleeping {sleep_for}s.", file=sys.stderr)
                    time.sleep(sleep_for)
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise RuntimeError("Too many retries after 503.")
                    continue

                resp.raise_for_status()
                xml_bytes = resp.content
                attempt = 0  # reset retries on success

                # get the server-supplied resumption token info for this response
                current_token_info = get_resumption_token_info(xml_bytes)
                current_next_token = current_token_info.get("text") or None

                # parse page and write records, but honor skip_until_identifier to avoid duplicates
                page_written = 0
                page_processed_records = 0

                for header, metadata in parse_records_from_xml(xml_bytes):
                    page_processed_records += 1
                    hdr_id = header.get("identifier")
                    hdr_date = header.get("datestamp")

                    # If we're still skipping until the checkpoint's last_identifier, do not write until we see it
                    if skip_until_identifier:
                        if hdr_id == skip_until_identifier:
                            # Found the last record we already have; from next record we resume writing
                            print(f"[INFO] Found last_identifier {skip_until_identifier} in stream; will resume writing after it.", file=sys.stderr)
                            skip_until_identifier = None
                            # do not write this record (it was already written)
                            continue
                        else:
                            # skip this record (we haven't reached the last_identifier yet)
                            continue

                    # write record
                    rec = {"oai_header": header, "metadata": metadata}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    page_written += 1
                    total_records += 1
                    total_records_written_this_run += 1
                    progress["records_written"] = total_records
                    progress["last_identifier"] = hdr_id
                    progress["last_datestamp"] = hdr_date

                    # periodic checkpoint while inside page (helps if page is large)
                    if total_records % CHECKPOINT_EVERY == 0:
                        save_checkpoint(current_next_token, current_token_info, out_path, progress)
                        print(f"[INFO] {total_records} records written. Checkpoint saved.", file=sys.stderr)

                # finished page: save a checkpoint
                save_checkpoint(current_next_token, current_token_info, out_path, progress)

                # examine progress / token behaviour to detect stuck loops
                if page_written == 0:
                    consecutive_no_progress += 1
                else:
                    consecutive_no_progress = 0

                next_token = get_resumption_token_from_xml(xml_bytes)
                # debug display of the token_info (safe)
                print(f"[DEBUG] Server resumptionToken info: {current_token_info}", file=sys.stderr)

                # Compare this next_token to previous_next_token to detect repeats
                if next_token == previous_next_token and next_token is not None:
                    consecutive_same_token += 1
                else:
                    consecutive_same_token = 0
                previous_next_token = next_token

                # If the server gives us a token, switch to token mode to follow it
                if next_token:
                    resumption_token = next_token
                    token_info = get_resumption_token_info(xml_bytes) or token_info
                    mode = "token"

                    # If the token appears to be expired immediately, and we have a last_datestamp, fallback
                    if token_is_expired(token_info):
                        if progress.get("last_datestamp"):
                            print("[INFO] Server returned a resumption token that is already expired. Falling back to resume-by-date.", file=sys.stderr)
                            mode = "from_date"
                            resumption_token = None
                            skip_until_identifier = progress.get("last_identifier")
                        else:
                            print("[WARN] Server returned a resumption token that is expired but no datestamp to fall back to. Continuing with token.", file=sys.stderr)

                else:
                    # no next_token -> harvest complete
                    print("[INFO] No resumptionToken found -> harvest complete for now.", file=sys.stderr)
                    break

                # If we've seen the same token repeat several times without page progress, assume token is stuck
                if (consecutive_same_token >= MAX_CONSECUTIVE_SAME_TOKEN) or (consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS):
                    # attempt safe fallback: if we have a last_datestamp, switch to date-resume and skip until last_identifier
                    cp = load_checkpoint()
                    last_dt = cp.get("progress", {}).get("last_datestamp") or progress.get("last_datestamp")
                    last_id = cp.get("progress", {}).get("last_identifier") or progress.get("last_identifier")
                    if last_dt:
                        print("[WARN] Detected stuck resumptionToken behavior (repeated token / no progress). Falling back to resume-by-date.", file=sys.stderr)
                        mode = "from_date"
                        resumption_token = None
                        token_info = {}
                        last_datestamp = last_dt
                        skip_until_identifier = last_id
                        # reset counters
                        consecutive_same_token = 0
                        consecutive_no_progress = 0
                        previous_next_token = None
                        # small polite sleep before continuing
                        time.sleep(2.0)
                        continue
                    else:
                        # no datestamp available -> give up to avoid infinite loop
                        print("[ERROR] Detected stuck token behavior but no last_datestamp available to fallback to. Aborting to avoid infinite loop.", file=sys.stderr)
                        break

                # polite sleep between pages
                time.sleep(INITIAL_SLEEP)

            except requests.RequestException as e:
                attempt += 1
                wait = (BACKOFF_FACTOR ** attempt)
                print(f"[ERROR] Request failed (attempt {attempt}): {e}", file=sys.stderr)
                # If we were using a resumptionToken and we've retried several times -> assume token is stale and fall back
                if attempt >= MAX_RETRIES and resumption_token:
                    cp = load_checkpoint()
                    last_dt = cp.get("progress", {}).get("last_datestamp") or last_datestamp
                    last_id = cp.get("progress", {}).get("last_identifier") or last_identifier
                    if last_dt:
                        print("[WARN] Retried with resumptionToken and it keeps failing. Falling back to resume-by-date using last_datestamp from checkpoint.", file=sys.stderr)
                        # switch to date resume and reset token
                        resumption_token = None
                        token_info = {}
                        mode = "from_date"
                        last_datestamp = last_dt
                        skip_until_identifier = last_id
                        attempt = 0
                        # reset detection counters
                        consecutive_same_token = 0
                        consecutive_no_progress = 0
                        previous_next_token = None
                        continue
                if attempt > MAX_RETRIES * 2:
                    print("[FATAL] Too many request failures; aborting.", file=sys.stderr)
                    traceback.print_exc()
                    break
                print(f"[INFO] Sleeping {wait:.1f}s before retrying...", file=sys.stderr)
                time.sleep(wait)
            except KeyboardInterrupt:
                print("[INFO] KeyboardInterrupt received. Saving checkpoint and exiting.", file=sys.stderr)
                save_checkpoint(resumption_token, token_info, out_path, progress)
                raise
            except Exception as e:
                print("[FATAL] Unexpected error:", e, file=sys.stderr)
                traceback.print_exc()
                save_checkpoint(resumption_token, token_info, out_path, progress)
                raise

    print(f"[DONE] Total records written this run: {total_records_written_this_run}  (cumulative total: {total_records})", file=sys.stderr)


if __name__ == "__main__":
    cp = load_checkpoint()
    start_token = cp.get("resumptionToken")
    start_token_info = cp.get("resumptionTokenInfo") or {}
    last_id = cp.get("progress", {}).get("last_identifier")
    last_dt = cp.get("progress", {}).get("last_datestamp")

    if start_token:
        if token_is_expired(start_token_info):
            print(f"[INFO] Checkpoint token expirationDate appears expired ({start_token_info.get('expirationDate')}). Will try fallback resume-by-date if possible.", file=sys.stderr)
        else:
            print(f"[INFO] Found checkpoint with resumptionToken: {start_token}", file=sys.stderr)
            print("[INFO] Will attempt to resume using that token. If it fails we'll automatically fall back to resume-by-date.", file=sys.stderr)

    harvest_all(out_path=OUTFILE,
                start_resumption_token=start_token,
                start_token_info=start_token_info,
                last_identifier=last_id,
                last_datestamp=last_dt)
