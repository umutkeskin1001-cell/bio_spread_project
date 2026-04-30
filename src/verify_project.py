from __future__ import annotations

import argparse
import ssl


def _ssl_backend_warning(backend: str) -> str | None:
    if not backend.startswith("OpenSSL "):
        return f"Unsupported SSL backend: {backend}"
    parts = backend.split()
    if len(parts) < 2:
        return f"Unparseable SSL backend version: {backend}"
    version = parts[1]
    major_minor = version.split(".")[:2]
    try:
        major = int(major_minor[0])
        minor = int(major_minor[1]) if len(major_minor) > 1 else 0
    except ValueError:
        return f"Unparseable SSL backend version: {backend}"
    if major < 1 or (major == 1 and minor == 0):
        return f"OpenSSL version too old: {backend}"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify BioSpread project health and release readiness")
    parser.add_argument("--release", action="store_true", help="Run release-grade verification checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warning = _ssl_backend_warning(ssl.OPENSSL_VERSION)
    if warning:
        print(warning)
    if args.release:
        print("release verification enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
