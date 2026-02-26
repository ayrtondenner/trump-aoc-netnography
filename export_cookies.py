"""
export_cookies.py - Export Twitter/X cookies from your browser
===============================================================
Extracts Twitter/X session cookies from your browser and saves them
in the format twikit expects (cookies.json).

Usage:
    python export_cookies.py

Prerequisites:
    - You must be logged into Twitter/X (x.com) in your browser
    - Close the browser before running (it locks the cookie database)

Tries Firefox (plaintext cookies), then Chrome, then Edge.
If all fail, prompts you to paste cookie values interactively.
"""

import base64
import ctypes
import ctypes.wintypes
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from config import COOKIES_PATH

# The cookies twikit needs to authenticate
REQUIRED_COOKIES = {"auth_token", "ct0"}
TWITTER_DOMAINS = {".x.com", ".twitter.com", "x.com", "twitter.com"}


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def dpapi_decrypt(encrypted: bytes) -> bytes:
    """Decrypt data using Windows DPAPI (CryptUnprotectData)."""
    blob_in = DATA_BLOB(len(encrypted), ctypes.cast(ctypes.create_string_buffer(encrypted, len(encrypted)), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()

    if ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raw = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)
        return raw
    raise OSError("DPAPI decryption failed")


def get_chrome_key() -> bytes | None:
    """Get Chrome's AES-GCM encryption key from Local State."""
    local_state_path = Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data" / "Local State"
    if not local_state_path.exists():
        return None

    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.load(f)

    encrypted_key_b64 = local_state.get("os_crypt", {}).get("encrypted_key")
    if not encrypted_key_b64:
        return None

    encrypted_key = base64.b64decode(encrypted_key_b64)
    # Remove "DPAPI" prefix (5 bytes)
    encrypted_key = encrypted_key[5:]
    return dpapi_decrypt(encrypted_key)


def decrypt_cookie_value(encrypted_value: bytes, key: bytes | None) -> str:
    """Decrypt a Chrome cookie value."""
    if not encrypted_value:
        return ""

    # v10 cookies: DPAPI encrypted (older Chrome)
    if encrypted_value[:3] == b"v10":
        try:
            return dpapi_decrypt(encrypted_value[3:]).decode("utf-8")
        except Exception:
            pass

    # v20 cookies: AES-256-GCM with app-bound key (Chrome 127+)
    if encrypted_value[:3] == b"v20" and key:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            # v20 format: "v20" (3) + nonce (12) + ciphertext + tag (16)
            nonce = encrypted_value[3:15]
            ciphertext = encrypted_value[15:]
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
        except ImportError:
            pass
        except Exception:
            pass

    # v10 with AES-GCM key (Chrome 80+)
    if key and len(encrypted_value) > 15:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            nonce = encrypted_value[3:15]
            ciphertext = encrypted_value[15:]
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
        except ImportError:
            pass
        except Exception:
            pass

    return ""


def extract_chrome_cookies() -> dict | None:
    """Extract Twitter cookies from Chrome's SQLite database."""
    chrome_cookie_path = Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data" / "Default" / "Network" / "Cookies"

    if not chrome_cookie_path.exists():
        # Try without "Network" subfolder (older Chrome)
        chrome_cookie_path = Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data" / "Default" / "Cookies"

    if not chrome_cookie_path.exists():
        print("  Chrome cookie database not found.")
        return None

    # Chrome locks the DB while running — copy to temp
    tmp = tempfile.mktemp(suffix=".db")
    try:
        shutil.copy2(str(chrome_cookie_path), tmp)
    except PermissionError:
        print("  Chrome cookie DB is locked. Please close Chrome and try again.")
        return None

    key = get_chrome_key()

    try:
        conn = sqlite3.connect(tmp)
        cursor = conn.cursor()

        # Query for Twitter/X domain cookies
        cursor.execute(
            "SELECT host_key, name, encrypted_value, value FROM cookies "
            "WHERE host_key LIKE '%x.com%' OR host_key LIKE '%twitter.com%'"
        )

        cookies = {}
        for host_key, name, encrypted_value, plain_value in cursor.fetchall():
            if not any(d in host_key for d in TWITTER_DOMAINS):
                continue

            # Try plain value first
            if plain_value:
                cookies[name] = plain_value
            elif encrypted_value:
                decrypted = decrypt_cookie_value(encrypted_value, key)
                if decrypted:
                    cookies[name] = decrypted

        conn.close()

        if REQUIRED_COOKIES.issubset(cookies.keys()):
            return cookies
        elif cookies:
            missing = REQUIRED_COOKIES - cookies.keys()
            print(f"  Found {len(cookies)} cookies but missing: {missing}")
            print(f"  (Cookie decryption may have failed for Chrome v127+ encrypted cookies)")
            return None
        else:
            print("  No Twitter cookies found in Chrome database.")
            return None

    except Exception as e:
        print(f"  Error reading Chrome cookies: {e}")
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def extract_edge_cookies() -> dict | None:
    """Extract Twitter cookies from Edge's SQLite database."""
    edge_cookie_path = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Edge" / "User Data" / "Default" / "Network" / "Cookies"

    if not edge_cookie_path.exists():
        edge_cookie_path = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Edge" / "User Data" / "Default" / "Cookies"

    if not edge_cookie_path.exists():
        print("  Edge cookie database not found.")
        return None

    # Edge uses same encryption as Chrome
    local_state_path = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Edge" / "User Data" / "Local State"
    key = None
    if local_state_path.exists():
        try:
            with open(local_state_path, "r", encoding="utf-8") as f:
                local_state = json.load(f)
            encrypted_key_b64 = local_state.get("os_crypt", {}).get("encrypted_key")
            if encrypted_key_b64:
                encrypted_key = base64.b64decode(encrypted_key_b64)[5:]
                key = dpapi_decrypt(encrypted_key)
        except Exception:
            pass

    tmp = tempfile.mktemp(suffix=".db")
    try:
        shutil.copy2(str(edge_cookie_path), tmp)
    except PermissionError:
        print("  Edge cookie DB is locked. Please close Edge and try again.")
        return None

    try:
        conn = sqlite3.connect(tmp)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT host_key, name, encrypted_value, value FROM cookies "
            "WHERE host_key LIKE '%x.com%' OR host_key LIKE '%twitter.com%'"
        )

        cookies = {}
        for host_key, name, encrypted_value, plain_value in cursor.fetchall():
            if not any(d in host_key for d in TWITTER_DOMAINS):
                continue
            if plain_value:
                cookies[name] = plain_value
            elif encrypted_value:
                decrypted = decrypt_cookie_value(encrypted_value, key)
                if decrypted:
                    cookies[name] = decrypted

        conn.close()

        if REQUIRED_COOKIES.issubset(cookies.keys()):
            return cookies
        elif cookies:
            missing = REQUIRED_COOKIES - cookies.keys()
            print(f"  Found {len(cookies)} cookies but missing: {missing}")
            return None
        else:
            print("  No Twitter cookies found in Edge database.")
            return None
    except Exception as e:
        print(f"  Error reading Edge cookies: {e}")
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def extract_firefox_cookies() -> dict | None:
    """Extract Twitter cookies from Firefox's SQLite database (plaintext, no encryption)."""
    profiles_dir = Path(os.environ.get("APPDATA", "")) / "Mozilla" / "Firefox" / "Profiles"

    if not profiles_dir.exists():
        print("  Firefox profiles directory not found.")
        return None

    # Find all profile folders with cookies.sqlite
    profile_dirs = [p for p in profiles_dir.iterdir() if p.is_dir()]
    if not profile_dirs:
        print("  No Firefox profiles found.")
        return None

    for profile_dir in profile_dirs:
        cookie_db = profile_dir / "cookies.sqlite"
        if not cookie_db.exists():
            continue

        print(f"  Checking profile: {profile_dir.name}")

        # Copy to temp (Firefox locks DB while running)
        tmp = tempfile.mktemp(suffix=".db")
        try:
            shutil.copy2(str(cookie_db), tmp)
        except PermissionError:
            print(f"  Firefox cookie DB is locked. Please close Firefox and try again.")
            return None

        try:
            conn = sqlite3.connect(tmp)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT host, name, value FROM moz_cookies "
                "WHERE host LIKE '%x.com%' OR host LIKE '%twitter.com%'"
            )

            cookies = {}
            for host, name, value in cursor.fetchall():
                if any(d in host for d in TWITTER_DOMAINS):
                    cookies[name] = value

            conn.close()

            if REQUIRED_COOKIES.issubset(cookies.keys()):
                print(f"  Found {len(cookies)} Twitter cookies in Firefox!")
                return cookies
            elif cookies:
                missing = REQUIRED_COOKIES - cookies.keys()
                print(f"  Found {len(cookies)} cookies but missing: {missing}")
            else:
                print(f"  No Twitter cookies in this profile.")

        except Exception as e:
            print(f"  Error reading Firefox cookies: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return None


def interactive_cookie_input() -> dict | None:
    """Prompt user to paste cookie values from browser DevTools."""
    print("""
============================================================
INTERACTIVE COOKIE ENTRY
============================================================

Open your browser, go to https://x.com (logged in), then:
  1. Press F12 to open DevTools
  2. Go to "Application" tab (Chrome/Edge) or "Storage" tab (Firefox)
  3. Click "Cookies" > "https://x.com"
  4. Find and copy the VALUE of: auth_token and ct0
============================================================
""")

    auth_token = input("Paste your auth_token value (or 'q' to quit): ").strip()
    if not auth_token or auth_token.lower() == "q":
        return None

    ct0 = input("Paste your ct0 value (or 'q' to quit): ").strip()
    if not ct0 or ct0.lower() == "q":
        return None

    cookies = {"auth_token": auth_token, "ct0": ct0}

    # Optionally collect extra cookies
    print("\nOptional (press Enter to skip):")
    for name in ["twid", "personalization_id", "guest_id"]:
        val = input(f"  {name}: ").strip()
        if val:
            cookies[name] = val

    return cookies


def main():
    print("=" * 60)
    print("Twitter/X Cookie Exporter")
    print("=" * 60)
    print()
    print("IMPORTANT: Close your browser before running this script.\n")

    cookies = None

    # Try Firefox first (plaintext cookies, most reliable)
    print("Trying Firefox...")
    cookies = extract_firefox_cookies()

    # Try Chrome
    if not cookies:
        print("\nTrying Chrome...")
        cookies = extract_chrome_cookies()

    # Try Edge
    if not cookies:
        print("\nTrying Edge...")
        cookies = extract_edge_cookies()

    # Interactive fallback
    if not cookies:
        print("\nAuto-extraction failed for all browsers.")
        cookies = interactive_cookie_input()

    if cookies:
        with open(COOKIES_PATH, "w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)

        print(f"\nCookies saved to {COOKIES_PATH}")
        print(f"Found {len(cookies)} cookies, including: {', '.join(sorted(REQUIRED_COOKIES & cookies.keys()))}")
        print("\nYou can now run: python 01_collect_data.py")
    else:
        print("\nNo cookies provided. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
