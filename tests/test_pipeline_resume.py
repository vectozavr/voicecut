from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


from voicecut.pipeline import stage


class AtomicResumeTests(unittest.TestCase):
    def test_resume_needs_marker_and_every_required_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "audio.wav"
            report = root / "report.json"
            logs = root / "logs"
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(output)!r}).write_bytes(b'wav'); "
                    f"Path({str(report)!r}).write_text('{{}}')"
                ),
            ]
            stage(
                "render",
                command,
                output=output,
                required_outputs=[report],
                resume=False,
                log_dir=logs,
            )
            report.unlink()
            stage(
                "render",
                command,
                output=output,
                required_outputs=[report],
                resume=True,
                log_dir=logs,
            )
            self.assertTrue(report.is_file())

    def test_child_resume_flag_does_not_invalidate_completed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            logs = root / "logs"
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(output)!r}).write_text('first')"
                ),
            ]
            stage(
                "asr",
                command,
                output=output,
                resume=False,
                log_dir=logs,
            )
            output.write_text("keep", encoding="utf-8")
            stage(
                "asr",
                [*command, "--resume"],
                output=output,
                resume=True,
                log_dir=logs,
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_failed_child_cannot_leave_a_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "partial.wav"
            logs = root / "logs"
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(output)!r}).write_bytes(b'partial'); "
                    "raise SystemExit(1)"
                ),
            ]
            with self.assertRaises(RuntimeError):
                stage(
                    "master",
                    command,
                    output=output,
                    resume=False,
                    log_dir=logs,
                )
            self.assertTrue(output.exists())
            self.assertFalse((logs / "completed" / "master.json").exists())


if __name__ == "__main__":
    unittest.main()
