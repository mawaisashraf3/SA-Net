import json
import subprocess
import sys


def test_installed_module_audit_command(config, records, tmp_path):
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sanet",
            "audit",
            "--manifest",
            config.data.manifest,
            "--root",
            config.data.root,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["records"] == 4
    assert json.loads(output.read_text())["dataset_sha256"] == report["dataset_sha256"]
