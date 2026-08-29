#!/usr/bin/env python3
import importlib.util
import shutil
import signal
import subprocess
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = REPOSITORY_ROOT / "scripts/run-cluster-stream.py"
SPEC = importlib.util.spec_from_file_location("run_cluster_stream", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher_module)


class ClusterStreamCleanupTest(unittest.TestCase):
    def setUp(self):
        args = launcher_module.parser().parse_args(["--no-viewer"])
        self.launcher = launcher_module.ClusterStreamLauncher(args)

    def tearDown(self):
        shutil.rmtree(self.launcher.runtime_dir, ignore_errors=True)

    def test_job_cancellation_waits_then_uses_pbs_force_delete(self):
        self.launcher.job_id = "12345.login01"
        self.launcher.job_active = True

        with mock.patch.object(
            self.launcher, "cleanup_remote", return_value=True
        ) as cleanup_remote:
            self.launcher.cancel_remote_job("interruption")

        command = cleanup_remote.call_args.args[0]
        self.assertIn('qdel "$job"', command)
        self.assertIn('qdel -W force "$job"', command)
        self.assertFalse(self.launcher.job_active)

    def test_ssh_master_isolated_from_terminal_interrupt_group(self):
        self.launcher.control_socket.touch()
        master = mock.Mock()
        master.poll.return_value = None
        ready = subprocess.CompletedProcess([], 0, "", "")

        with mock.patch.object(
            launcher_module.subprocess, "Popen", return_value=master
        ) as popen, mock.patch.object(
            launcher_module.subprocess, "run", return_value=ready
        ):
            self.launcher.start_master()

        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_mac_wake_guard_allows_display_sleep(self):
        wake_guard = mock.Mock()
        wake_guard.poll.return_value = None

        with mock.patch.object(
            launcher_module.sys, "platform", "darwin"
        ), mock.patch.object(
            launcher_module.subprocess, "Popen", return_value=wake_guard
        ) as popen, mock.patch.object(
            launcher_module.time, "sleep"
        ):
            self.launcher.start_wake_guard()

        self.assertEqual(
            popen.call_args.args[0],
            ["caffeinate", "-i", "-w", str(launcher_module.os.getpid())],
        )
        self.assertNotIn("-d", popen.call_args.args[0])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_remote_cleanup_reconnects_when_master_command_fails(self):
        self.launcher.control_socket.touch()
        self.launcher.master = mock.Mock()
        self.launcher.master.poll.return_value = None
        failed_master = subprocess.CompletedProcess([], 255, "", "lost")
        successful_direct = subprocess.CompletedProcess([], 0, "", "")

        with mock.patch.object(
            self.launcher, "remote", return_value=failed_master
        ) as master_remote, mock.patch.object(
            launcher_module.subprocess, "run", return_value=successful_direct
        ) as direct_remote:
            self.assertTrue(self.launcher.cleanup_remote("true"))

        master_remote.assert_called_once()
        direct_command = direct_remote.call_args.args[0]
        self.assertEqual(direct_command[0], "ssh")
        self.assertIn("ConnectTimeout=15", direct_command)

    def test_cleanup_keeps_viewer_alive_and_stops_stream_before_remote_resources(self):
        self.launcher.client = mock.Mock(name="client")
        self.launcher.viewer = mock.Mock(name="viewer")
        self.launcher.wake_guard = mock.Mock(name="wake_guard")
        self.launcher.stream_state_started = True
        original_sigint = signal.getsignal(signal.SIGINT)
        calls = []

        with mock.patch.object(
            self.launcher,
            "stop_process",
            side_effect=lambda process, **kwargs: calls.append(
                "client"
                if process is self.launcher.client
                else "wake"
                if process is self.launcher.wake_guard
                else "viewer"
            ),
        ), mock.patch.object(
            self.launcher,
            "publish_stream_state",
            side_effect=lambda status: calls.append(status),
        ), mock.patch.object(
            self.launcher,
            "cancel_remote_job",
            side_effect=lambda reason: calls.append("job"),
        ), mock.patch.object(
            self.launcher,
            "stop_remote_relay",
            side_effect=lambda: calls.append("relay"),
        ), mock.patch.object(
            self.launcher,
            "stop_tunnel",
            side_effect=lambda: calls.append("tunnel"),
        ), mock.patch.object(
            self.launcher,
            "close_master",
            side_effect=lambda: calls.append("master"),
        ):
            self.launcher.cleanup(interrupted=True)

        self.assertEqual(
            calls,
            ["client", "ended", "job", "relay", "tunnel", "master", "wake"],
        )
        self.assertEqual(signal.getsignal(signal.SIGINT), original_sigint)
        self.assertFalse(self.launcher.runtime_dir.exists())

    def test_stream_state_is_atomic_and_new_run_hides_stale_latest_frame(self):
        output_dir = self.launcher.runtime_dir / "frames"
        output_dir.mkdir()
        (output_dir / ".nbody-latest").write_text("old.nbsnap\n")
        self.launcher.args.output_dir = str(output_dir)

        self.launcher.publish_stream_state("active")

        self.assertFalse((output_dir / ".nbody-latest").exists())
        self.assertEqual(
            (output_dir / ".nbody-stream-state").read_text(),
            f"active {self.launcher.run_id}\n",
        )
        self.assertTrue(self.launcher.stream_state_started)

        self.launcher.publish_stream_state("ended")
        self.assertEqual(
            (output_dir / ".nbody-stream-state").read_text(),
            f"ended {self.launcher.run_id}\n",
        )

    def test_job_writes_live_logs_and_requests_a_bounded_stream_drain(self):
        script = self.launcher.job_script()

        self.assertIn("--stream-drain-ms 30000", script)
        self.assertIn(
            f">{self.launcher.remote_run_dir}/simulation.out", script
        )
        self.assertIn(
            f"2>{self.launcher.remote_run_dir}/simulation.err", script
        )
        self.assertIn(
            f">{self.launcher.remote_run_dir}/simulation.exit-status", script
        )
        qsub = self.launcher.qsub_arguments()
        self.assertIn(f"{self.launcher.remote_run_dir}/pbs.out", qsub)
        self.assertIn(f"{self.launcher.remote_run_dir}/pbs.err", qsub)

    def test_nonzero_simulation_exit_status_is_a_launcher_failure(self):
        result = subprocess.CompletedProcess([], 0, "2\n", "")

        with mock.patch.object(self.launcher, "remote", return_value=result):
            with self.assertRaisesRegex(
                launcher_module.LaunchError, "simulation exited with status 2"
            ):
                self.launcher.require_successful_simulation()

    def test_missing_simulation_exit_status_is_a_launcher_failure(self):
        result = subprocess.CompletedProcess([], 1, "", "")

        with mock.patch.object(self.launcher, "remote", return_value=result):
            with self.assertRaisesRegex(
                launcher_module.LaunchError, "without a readable simulation exit status"
            ):
                self.launcher.require_successful_simulation()


if __name__ == "__main__":
    unittest.main()
