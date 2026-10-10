"""Download the original GloVe 6B archive and the required BERT checkpoint."""
import argparse
import hashlib
import json
import os
import io
from pathlib import Path
import time
import zipfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import requests
from huggingface_hub import snapshot_download, configure_http_backend

ROOT = Path(__file__).resolve().parent
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
GLOVE_SHA256 = "95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544"


class RemoteZip(io.RawIOBase):
    """Read a ZIP from HTTP byte ranges without downloading its other members."""
    def __init__(self, session, url):
        self.session, self.url, self.position = session, url, 0
        response = session.get(url, headers={"Range": "bytes=0-0"}, timeout=(15, 30))
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError("The download server does not support byte ranges")
        self.size = int(response.headers["Content-Range"].split("/")[-1])
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        count = min(count, 4 * 1024 * 1024)
        for attempt in range(3):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={self.position}-{self.position + count - 1}"},
                                            timeout=(15, 60))
                response.raise_for_status()
                block = response.content
                if response.status_code != 206 or len(block) != count:
                    raise ValueError("Invalid HTTP range response")
                self.position += len(block)
                return block
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["all", "glove", "bert"], default="all")
    parser.add_argument("--direct", action="store_true", help="Ignore stale OS proxy settings for downloads")
    parser.add_argument("--glove-range", action="store_true", help="Read only the 100d member of the original ZIP using byte ranges")
    parser.add_argument("--bert-source", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()
    def session_factory():
        session = requests.Session()
        if args.direct:
            session.trust_env = False
        return session
    configure_http_backend(backend_factory=session_factory)
    session = session_factory()
    cache = ROOT / "assets"
    cache.mkdir(exist_ok=True)
    metadata_path = cache / "sources.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if args.asset in ("all", "glove"):
        target = cache / "glove.6B.100d.txt"
        archive = cache / "glove.6B.zip"
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        if not target.exists():
            if args.glove_range:
                partial_text = target.with_suffix(".txt.part")
                with zipfile.ZipFile(RemoteZip(session, url)) as zipped, zipped.open(target.name) as source, partial_text.open("wb") as output:
                    total = last = 0
                    while block := source.read(16 * 1024 * 1024):
                        output.write(block)
                        total += len(block)
                        if total - last >= 50 * 1024 * 1024:
                            print(f"GloVe extracted {total / 1024**2:.0f} MiB", flush=True)
                            last = total
                partial_text.replace(target)
            else:
                download_glove_archive(session, url, archive, cache)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if digest != GLOVE_SHA256:
            raise ValueError("GloVe checksum differs from the original 6B 100-dimensional file")
        metadata["glove"] = {"url": url, "file": target.name, "sha256": digest}
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print("GloVe ready", flush=True)
    if args.asset in ("all", "bert"):
        model_id = "google-bert/bert-base-uncased"
        revision = metadata.get("bert", {}).get("revision") or BERT_REVISION
        metadata["bert"] = {"model_id": model_id, "revision": revision}
        metadata_path.write_text(json.dumps(metadata, indent=2))
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
        folder = cache / "bert-base-uncased"
        if args.bert_source == "huggingface":
            snapshot_download(model_id, revision=revision, local_dir=folder, allow_patterns=files + ["model.safetensors"])
        else:
            if not all((folder / name).exists() for name in files[:-1]):
                snapshot_download(model_id, revision=revision, local_dir=folder, allow_patterns=files)
            url = "https://modelscope.cn/api/v1/models/AI-ModelScope/bert-base-uncased/repo?Revision=master&FilePath=model.safetensors"
            target = folder / "model.safetensors"
            if not target.exists():
                partial = target.with_suffix(".safetensors.part")
                with session.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        total = last = 0
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                            total += len(block)
                            if total - last >= 50 * 1024 * 1024:
                                print(f"BERT download {total / 1024**2:.0f} MiB", flush=True)
                                last = total
                digest = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
                if digest != BERT_SHA256:
                    raise ValueError("Mirror weight checksum differs from the pinned official checkpoint")
                partial.replace(target)
            metadata["bert"]["delivery_url"] = url
        digest = hashlib.file_digest((folder / "model.safetensors").open("rb"), "sha256").hexdigest()
        if digest != BERT_SHA256:
            raise ValueError("BERT checksum differs from the pinned official checkpoint")
        metadata["bert"]["sha256"] = digest
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print("BERT ready", flush=True)


def download_glove_archive(session, url, archive, cache):
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                total = last = 0
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
                    total += len(block)
                    if total - last >= 50 * 1024 * 1024:
                        print(f"GloVe download {total / 1024**2:.0f} MiB", flush=True)
                        last = total
        partial.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extract("glove.6B.100d.txt", cache)


if __name__ == "__main__":
    main()
