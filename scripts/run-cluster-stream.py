#!/usr/bin/env python3
"""Run a cluster simulation and its live Mac stream with one command."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Optional, Sequence


DEFAULT_SSH_TARGET = "u10774182@10.78.18.100"
DEFAULT_REMOTE_PROJECT = "/home/u10774182/AMSC-Nbody-fresh"
DEFAULT_CONTAINER = (
    "/home/u10774182/AMSC-Nbody-ci/releases/"
    "a1ae8c658170308a75c636631a57893a67b2d08a/nbody-tests.sif"
)


class LaunchError(RuntimeError):
    """A user-facing launcher failure."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def port(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def remote_path(project: str, value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        return str(path)
    return str(PurePosixPath(project) / path)


def shell_join(arguments: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(argument)) for argument in arguments)


def parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parent.parent
    result = argparse.ArgumentParser(
        description=(
            "Start the login-node relay, SSH tunnel, Mac client, Metal viewer, "
            "and a PBS GPU simulation using one multiplexed SSH login."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    connection = result.add_argument_group("cluster connection")
    connection.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    connection.add_argument("--remote-project", default=DEFAULT_REMOTE_PROJECT)
    connection.add_argument("--container", default=DEFAULT_CONTAINER)
    connection.add_argument("--queue", default="gpu")
    connection.add_argument(
        "--qsub-arg",
        action="append",
        default=[],
        help="extra qsub argument; repeat and use --qsub-arg=-l for leading dashes",
    )

    simulation = result.add_argument_group("simulation")
    simulation.add_argument("--input", default="data/two_body.bin")
    simulation.add_argument("--steps", type=positive_int, default=1_000_000)
    simulation.add_argument("--time-step", type=positive_float, default=0.001)
    simulation.add_argument("--theta", type=positive_float, default=0.5)
    simulation.add_argument("--softening", type=positive_float, default=1.0e-6)
    simulation.add_argument("--sample-rate", type=positive_float, default=60.0)
    simulation.add_argument(
        "--stream-max-particles", type=nonnegative_int, default=100_000
    )
    simulation.add_argument(
        "--sim-arg",
        action="append",
        default=[],
        help="extra simulator argument; repeat and use --sim-arg=--flag",
    )
    simulation.add_argument(
        "--profile-stages",
        action="store_true",
        help="print timing for the simulator initialization and step stages",
    )

    streaming = result.add_argument_group("streaming")
    streaming.add_argument("--spool-dir", default="nbody-frames-live")
    streaming.add_argument(
        "--output-dir", default=str(project / "received-frames")
    )
    streaming.add_argument("--relay-host", default="login01")
    streaming.add_argument("--relay-source-port", type=port, default=4748)
    streaming.add_argument("--relay-client-port", type=port, default=4747)
    streaming.add_argument("--local-port", type=port, default=4747)
    streaming.add_argument("--reconnect-ms", type=positive_int, default=1000)
    streaming.add_argument(
        "--stream-drain-seconds",
        type=positive_float,
        default=30.0,
        help="maximum time the compute source waits for final frame acknowledgements",
    )
    streaming.add_argument(
        "--no-viewer",
        action="store_true",
        help="download frames without opening the local Metal viewer",
    )
    streaming.add_argument(
        "--viewer-poll-ms",
        type=positive_float,
        default=16.0,
        help="how often the Metal viewer checks for a newly completed frame",
    )
    streaming.add_argument(
        "--viewer-replay-fps",
        type=positive_float,
        default=30.0,
        help="play a completed stream in a loop at this frame rate",
    )
    streaming.add_argument(
        "--close-viewer-on-exit",
        action="store_true",
        help="close the Metal viewer when the launcher finishes",
    )

    lifecycle = result.add_argument_group("lifecycle")
    lifecycle.add_argument(
        "--post-job-wait",
        type=float,
        default=2.0,
        help="seconds to leave the client connected after the PBS job ends",
    )
    lifecycle.add_argument(
        "--leave-job-running",
        action="store_true",
        help="do not qdel an active job when the launcher is interrupted",
    )
    lifecycle.add_argument(
        "--dry-run",
        action="store_true",
        help="print the generated PBS job without connecting",
    )
    return result


class ClusterStreamLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.local_project = Path(__file__).resolve().parent.parent
        self.client_script = self.local_project / "scripts/run-stream-client.sh"
        self.viewer_build_dir = self.local_project / "build/viewer"
        self.viewer_executable = self.viewer_build_dir / "nbody_viewer"
        self.remote_project = str(PurePosixPath(args.remote_project))
        self.remote_input = remote_path(self.remote_project, args.input)
        self.remote_spool = remote_path(self.remote_project, args.spool_dir)
        self.remote_token = str(PurePosixPath(self.remote_project) / ".nbody-stream-token")
        self.remote_simulator = str(
            PurePosixPath(self.remote_project) / "build/cluster-relay/nbody"
        )
        self.remote_relay = str(
            PurePosixPath(self.remote_project)
            / "build/cluster-relay/nbody_stream_relay"
        )
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}-{os.getpid()}"
        self.remote_run_dir = str(
            PurePosixPath(self.remote_project) / ".nbody-stream-runs" / self.run_id
        )
        self.runtime_dir = Path(tempfile.mkdtemp(prefix="nbody-stream-"))
        self.control_socket = self.runtime_dir / "ssh-control"
        self.wake_guard: Optional[subprocess.Popen[bytes]] = None
        self.master: Optional[subprocess.Popen[bytes]] = None
        self.tunnel_active = False
        self.client: Optional[subprocess.Popen[bytes]] = None
        self.viewer: Optional[subprocess.Popen[bytes]] = None
        self.viewer_pid: Optional[int] = None
        self.relay_started = False
        self.job_id: Optional[str] = None
        self.job_active = False
        self.cleanup_started = False
        self.stream_state_started = False

    @staticmethod
    def status(message: str) -> None:
        print(f"[nbody] {message}", flush=True)

    def ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-S",
            str(self.control_socket),
            "-o",
            "ControlMaster=no",
            self.args.ssh_target,
        ]

    def remote(
        self,
        command: str,
        *,
        input_text: Optional[str] = None,
        check: bool = True,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self.ssh_base() + [f"bash -lc {shlex.quote(command)}"],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise LaunchError(
                f"cluster command failed ({result.returncode}): {details}"
            )
        return result

    def cleanup_remote(self, command: str, *, timeout: float = 45.0) -> bool:
        """Run cleanup through the master, reconnecting directly if it died."""
        if (
            self.master is not None
            and self.master.poll() is None
            and self.control_socket.exists()
        ):
            try:
                result = self.remote(command, check=False, timeout=timeout)
                if result.returncode == 0:
                    return True
                details = result.stderr.strip() or result.stdout.strip()
                self.status(
                    "existing SSH connection could not finish cleanup"
                    + (f": {details}" if details else "")
                )
            except subprocess.TimeoutExpired:
                self.status("existing SSH connection timed out during cleanup")

        self.status("reconnecting to login01 for final remote cleanup")
        direct_command = [
            "ssh",
            "-o",
            "ControlMaster=no",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            self.args.ssh_target,
            f"bash -lc {shlex.quote(command)}",
        ]
        try:
            result = subprocess.run(
                direct_command,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            self.status("WARNING: direct SSH cleanup timed out")
            return False
        if result.returncode == 0:
            return True
        details = result.stderr.strip() or result.stdout.strip()
        self.status(
            "WARNING: direct SSH cleanup failed"
            + (f": {details}" if details else "")
        )
        return False

    def simulation_arguments(self) -> list[str]:
        arguments = [
            self.remote_simulator,
            "--input",
            self.remote_input,
            "--time-step",
            str(self.args.time_step),
            "--steps",
            str(self.args.steps),
            "--theta",
            str(self.args.theta),
            "--softening",
            str(self.args.softening),
            "--cluster",
            "--stream-dir",
            self.remote_spool,
            "--sample-rate",
            str(self.args.sample_rate),
            "--stream-max-particles",
            str(self.args.stream_max_particles),
            "--stream-relay-host",
            self.args.relay_host,
            "--stream-relay-port",
            str(self.args.relay_source_port),
            "--stream-relay-token-file",
            self.remote_token,
            "--stream-reconnect-ms",
            str(self.args.reconnect_ms),
            "--stream-drain-ms",
            str(max(1, round(self.args.stream_drain_seconds * 1000.0))),
        ]
        if self.args.profile_stages:
            arguments.append("--profile-stages")
        arguments.extend(self.args.sim_arg)
        return arguments

    def job_script(self) -> str:
        simulation = shell_join(self.simulation_arguments())
        stdout_path = str(PurePosixPath(self.remote_run_dir) / "simulation.out")
        stderr_path = str(PurePosixPath(self.remote_run_dir) / "simulation.err")
        exit_status_path = str(
            PurePosixPath(self.remote_run_dir) / "simulation.exit-status"
        )
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"cd {shlex.quote(self.remote_project)}",
                f"mkdir -p {shlex.quote(self.remote_spool)} {shlex.quote(self.remote_run_dir)}",
                "set +e",
                shell_join(["apptainer", "exec", "--nv", self.args.container])
                + " "
                + simulation
                + f" >{shlex.quote(stdout_path)} 2>{shlex.quote(stderr_path)}",
                "simulation_status=$?",
                f"printf '%s\\n' \"$simulation_status\" >{shlex.quote(exit_status_path)}",
                "exit \"$simulation_status\"",
                "",
            ]
        )

    def qsub_arguments(self) -> list[str]:
        stdout_path = str(PurePosixPath(self.remote_run_dir) / "pbs.out")
        stderr_path = str(PurePosixPath(self.remote_run_dir) / "pbs.err")
        return [
            "qsub",
            "-q",
            self.args.queue,
            "-N",
            "nbody_stream",
            "-o",
            stdout_path,
            "-e",
            stderr_path,
            *self.args.qsub_arg,
        ]

    def print_dry_run(self) -> None:
        print(f"SSH target: {self.args.ssh_target}")
        print(f"Remote run directory: {self.remote_run_dir}")
        print(f"Local output directory: {Path(self.args.output_dir).expanduser()}")
        print("PBS submission command:")
        print(shell_join(self.qsub_arguments()))
        print("PBS job script:")
        print(self.job_script(), end="")

    def check_local_prerequisites(self) -> None:
        if shutil.which("ssh") is None:
            raise LaunchError("ssh is not installed or is not on PATH")
        if sys.platform == "darwin" and shutil.which("caffeinate") is None:
            raise LaunchError("caffeinate is required to keep the Mac awake")
        if not self.client_script.is_file():
            raise LaunchError(f"client launcher not found: {self.client_script}")
        if not self.args.no_viewer:
            if sys.platform != "darwin":
                raise LaunchError(
                    "the live viewer requires macOS; use --no-viewer on other systems"
                )
            if shutil.which("cmake") is None:
                raise LaunchError("cmake is required to build the Metal viewer")
        output_dir = Path(self.args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.args.local_port))
            except OSError as error:
                raise LaunchError(
                    f"local port {self.args.local_port} is already in use"
                ) from error

    def start_wake_guard(self) -> None:
        if sys.platform != "darwin":
            return
        self.wake_guard = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(0.1)
        if self.wake_guard.poll() is not None:
            raise LaunchError("could not prevent idle system sleep")
        self.status(
            "Mac idle sleep disabled for this run; display sleep remains enabled"
        )

    def build_viewer(self) -> None:
        if self.args.no_viewer:
            return
        configure = [
            "cmake",
            "-S",
            str(self.local_project),
            "-B",
            str(self.viewer_build_dir),
            "-DBUILD_TESTING=OFF",
            "-DNBODY_ENABLE_CUDA=OFF",
            "-DNBODY_BUILD_SIMULATION=OFF",
            "-DNBODY_BUILD_STREAMING=OFF",
            "-DNBODY_BUILD_VIEWER=ON",
        ]
        if (
            not (self.viewer_build_dir / "CMakeCache.txt").exists()
            and shutil.which("ninja") is not None
        ):
            configure.extend(["-G", "Ninja"])

        self.status("building the local Metal viewer")
        try:
            subprocess.run(
                configure,
                cwd=self.local_project,
                check=True,
            )
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(self.viewer_build_dir),
                    "--target",
                    "nbody_viewer",
                    "--parallel",
                ],
                cwd=self.local_project,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise LaunchError("the local Metal viewer failed to build") from error
        if not os.access(self.viewer_executable, os.X_OK):
            raise LaunchError(
                f"viewer executable was not created: {self.viewer_executable}"
            )

    def start_master(self) -> None:
        self.status(
            f"opening SSH connection to {self.args.ssh_target}; authenticate if prompted"
        )
        command = [
            "ssh",
            "-M",
            "-S",
            str(self.control_socket),
            "-o",
            "ControlMaster=yes",
            "-o",
            "ControlPersist=no",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            self.args.ssh_target,
        ]
        # Keep the multiplexed SSH transport out of the terminal foreground
        # process group. Ctrl+C should interrupt the launcher, not destroy the
        # connection that the launcher's cleanup needs for qdel and relay stop.
        self.master = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                raise LaunchError("SSH master connection ended during authentication")
            if self.control_socket.exists():
                check = subprocess.run(
                    [
                        "ssh",
                        "-S",
                        str(self.control_socket),
                        "-O",
                        "check",
                        self.args.ssh_target,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if check.returncode == 0:
                    self.status("SSH connection ready; subsequent operations reuse it")
                    return
            time.sleep(0.2)
        raise LaunchError("timed out waiting for SSH authentication")

    def preflight_cluster(self) -> None:
        required_executables = [self.remote_simulator, self.remote_relay]
        required_files = [self.remote_token, self.remote_input, self.args.container]
        checks = ["set -eu", f"cd {shlex.quote(self.remote_project)}"]
        checks.extend(f"test -x {shlex.quote(path)}" for path in required_executables)
        checks.extend(f"test -r {shlex.quote(path)}" for path in required_files)
        checks.extend(["command -v qsub >/dev/null", "command -v apptainer >/dev/null"])
        result = self.remote("; ".join(checks), check=False)
        if result.returncode != 0:
            raise LaunchError(
                "cluster preflight failed; check the input, container, token, and "
                "build/cluster-relay executables"
            )
        self.status("cluster input, token, container, and executables verified")

    def start_relay(self) -> None:
        relay_log = str(PurePosixPath(self.remote_run_dir) / "relay.log")
        relay_pid = str(PurePosixPath(self.remote_run_dir) / "relay.pid")
        relay_command = shell_join(
            [
                self.remote_relay,
                "--token-file",
                self.remote_token,
                "--source-bind",
                "0.0.0.0",
                "--source-port",
                self.args.relay_source_port,
                "--client-bind",
                "127.0.0.1",
                "--client-port",
                self.args.relay_client_port,
            ]
        )
        command = "\n".join(
            [
                "set -eu",
                f"mkdir -p {shlex.quote(self.remote_run_dir)}",
                f"nohup {relay_command} >{shlex.quote(relay_log)} 2>&1 </dev/null &",
                "relay_pid=$!",
                f"printf '%s\\n' \"$relay_pid\" >{shlex.quote(relay_pid)}",
                "sleep 1",
                "kill -0 \"$relay_pid\"",
            ]
        )
        result = self.remote(command, check=False)
        if result.returncode != 0:
            log = self.remote(
                f"sed -n '1,80p' {shlex.quote(relay_log)}", check=False
            ).stdout.strip()
            raise LaunchError(f"relay failed to start: {log or 'no relay log output'}")
        self.relay_started = True
        self.status(
            f"relay ready on cluster ports {self.args.relay_source_port}/"
            f"{self.args.relay_client_port}"
        )

    def start_tunnel(self) -> None:
        forwarding = self.forwarding_spec()
        command = [
            "ssh",
            "-S",
            str(self.control_socket),
            "-O",
            "forward",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            forwarding,
            self.args.ssh_target,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise LaunchError(f"SSH tunnel failed to start: {details}")
        self.tunnel_active = True
        self.status(f"SSH tunnel ready on 127.0.0.1:{self.args.local_port}")

    def forwarding_spec(self) -> str:
        return (
            f"127.0.0.1:{self.args.local_port}:"
            f"127.0.0.1:{self.args.relay_client_port}"
        )

    def start_client(self) -> None:
        self.publish_stream_state("active")
        environment = os.environ.copy()
        environment.update(
            {
                "NBODY_STREAM_HOST": "127.0.0.1",
                "NBODY_STREAM_PORT": str(self.args.local_port),
                "NBODY_FRAME_OUTPUT_DIR": str(
                    Path(self.args.output_dir).expanduser().resolve()
                ),
            }
        )
        self.client = subprocess.Popen(
            [str(self.client_script)],
            cwd=self.local_project,
            env=environment,
            start_new_session=True,
        )
        time.sleep(0.5)
        if self.client.poll() is not None:
            raise LaunchError("Mac streaming client exited during startup")
        self.status(f"Mac client writing frames to {self.args.output_dir}")

    def publish_stream_state(self, status: str) -> None:
        if status not in {"active", "ended"}:
            raise ValueError(f"invalid stream state: {status}")
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / ".nbody-stream-state"
        temporary = output_dir / f".nbody-stream-state.{os.getpid()}.part"
        temporary.write_text(f"{status} {self.run_id}\n", encoding="utf-8")
        if status == "active":
            (output_dir / ".nbody-latest").unlink(missing_ok=True)
        os.replace(temporary, marker)
        if status == "active":
            self.stream_state_started = True

    def existing_viewer_pid(self, output_dir: Path) -> Optional[int]:
        pid_file = output_dir / ".nbody-viewer.pid"
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                check=False,
            )
            command = result.stdout.strip()
            if (
                result.returncode == 0
                and str(self.viewer_executable) in command
                and str(output_dir) in command
            ):
                return pid
        except (OSError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)
        return None

    def start_viewer(self) -> None:
        if self.args.no_viewer:
            return
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        existing_pid = self.existing_viewer_pid(output_dir)
        if existing_pid is not None:
            self.viewer_pid = existing_pid
            self.status(
                f"reusing Metal viewer {existing_pid}; it is waiting for this run"
            )
            return

        log_path = output_dir / ".nbody-viewer.log"
        with log_path.open("ab", buffering=0) as viewer_log:
            self.viewer = subprocess.Popen(
                [
                    str(self.viewer_executable),
                    "--stream-dir",
                    str(output_dir),
                    "--poll-ms",
                    str(self.args.viewer_poll_ms),
                    "--replay-fps",
                    str(self.args.viewer_replay_fps),
                ],
                cwd=self.local_project,
                stdout=viewer_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        time.sleep(0.5)
        if self.viewer.poll() is not None:
            raise LaunchError("Metal viewer exited during startup")
        self.viewer_pid = self.viewer.pid
        pid_file = output_dir / ".nbody-viewer.pid"
        temporary = output_dir / f".nbody-viewer.pid.{os.getpid()}.part"
        temporary.write_text(f"{self.viewer.pid}\n", encoding="utf-8")
        os.replace(temporary, pid_file)
        self.status("Metal viewer opened and is waiting for live frames")

    def submit_job(self) -> None:
        qsub = shell_join(self.qsub_arguments())
        result = self.remote(qsub, input_text=self.job_script(), check=False)
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise LaunchError(f"qsub failed: {details}")
        matches = re.findall(r"(?m)^\s*([0-9]+(?:\.[A-Za-z0-9_.-]+)?)\s*$", result.stdout)
        if not matches:
            raise LaunchError(f"could not read the PBS job ID from: {result.stdout!r}")
        self.job_id = matches[-1]
        self.job_active = True
        job_file = str(PurePosixPath(self.remote_run_dir) / "job-id")
        self.remote(
            f"printf '%s\\n' {shlex.quote(self.job_id)} >{shlex.quote(job_file)}"
        )
        self.status(f"submitted PBS job {self.job_id}")
        self.status(f"remote logs: {self.remote_run_dir}")

    def monitor_job(self) -> None:
        assert self.job_id is not None
        previous_state = ""
        while True:
            if self.master is not None and self.master.poll() is not None:
                raise LaunchError("SSH connection was lost")
            if self.client is not None and self.client.poll() is not None:
                raise LaunchError("Mac streaming client stopped unexpectedly")
            result = self.remote(
                f"qstat -f {shlex.quote(self.job_id)}", check=False
            )
            if result.returncode == 255:
                raise LaunchError("SSH connection failed while monitoring PBS")
            if result.returncode != 0:
                self.job_active = False
                self.status(f"PBS job {self.job_id} has left the queue")
                break
            state_match = re.search(r"job_state\s*=\s*([A-Z])", result.stdout)
            state = state_match.group(1) if state_match else "?"
            if state in {"C", "F"}:
                self.job_active = False
                self.status(f"PBS job {self.job_id} completed")
                break
            if state != previous_state:
                names = {
                    "Q": "queued",
                    "R": "running",
                    "E": "exiting",
                    "H": "held",
                    "C": "complete",
                    "F": "finished",
                }
                self.status(f"PBS job state: {names.get(state, state)}")
                previous_state = state
            time.sleep(2.0)
        if self.args.post_job_wait > 0:
            self.status(
                f"leaving the client connected for {self.args.post_job_wait:g} seconds"
            )
            time.sleep(self.args.post_job_wait)

    def show_job_tail(self) -> None:
        stdout_path = str(PurePosixPath(self.remote_run_dir) / "simulation.out")
        stderr_path = str(PurePosixPath(self.remote_run_dir) / "simulation.err")
        result = self.remote(
            "\n".join(
                [
                    f"test ! -f {shlex.quote(stdout_path)} || tail -n 12 {shlex.quote(stdout_path)}",
                    f"test ! -s {shlex.quote(stderr_path)} || {{ echo '--- simulation stderr ---'; "
                    f"head -n 1 {shlex.quote(stderr_path)}; "
                    f"lines=$(wc -l <{shlex.quote(stderr_path)}); "
                    f"if [ \"$lines\" -gt 13 ]; then echo '...'; fi; "
                    f"tail -n 12 {shlex.quote(stderr_path)}; }}",
                ]
            ),
            check=False,
        )
        if result.stdout.strip():
            self.status("final simulation output:")
            print(result.stdout.rstrip())

    def require_successful_simulation(self) -> None:
        exit_status_path = str(
            PurePosixPath(self.remote_run_dir) / "simulation.exit-status"
        )
        result = self.remote(
            f"test -r {shlex.quote(exit_status_path)} && cat {shlex.quote(exit_status_path)}",
            check=False,
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not re.fullmatch(r"[0-9]+", value):
            raise LaunchError(
                "PBS job ended without a readable simulation exit status; "
                f"inspect {self.remote_run_dir}"
            )
        if int(value) != 0:
            raise LaunchError(
                f"simulation exited with status {value}; inspect "
                f"{self.remote_run_dir}/simulation.err"
            )

    @staticmethod
    def stop_process(
        process: Optional[subprocess.Popen[bytes]], *, process_group: bool = False
    ) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            if process_group:
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            process.wait(timeout=3.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            try:
                if process_group:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def cancel_remote_job(self, reason: str) -> None:
        if not self.job_active or self.job_id is None:
            return
        if self.args.leave_job_running:
            self.status(f"leaving PBS job {self.job_id} running as requested")
            return

        job = shlex.quote(self.job_id)
        self.status(f"cancelling PBS job {self.job_id} after {reason}")
        command = "\n".join(
            [
                "set +e",
                f"job={job}",
                "qstat -f \"$job\" >/dev/null 2>&1 || exit 0",
                "qdel \"$job\" >/dev/null 2>&1",
                "for attempt in 1 2 3 4 5; do",
                "  qstat -f \"$job\" >/dev/null 2>&1 || exit 0",
                "  sleep 1",
                "done",
                "qdel -W force \"$job\" >/dev/null 2>&1",
                "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do",
                "  qstat -f \"$job\" >/dev/null 2>&1 || exit 0",
                "  sleep 1",
                "done",
                "state=$(qstat -f \"$job\" 2>/dev/null | "
                "awk -F= '/job_state/{gsub(/[[:space:]]/, \"\", $2); print $2; exit}')",
                "case \"$state\" in E|C|F) exit 0;; *) exit 1;; esac",
            ]
        )
        if self.cleanup_remote(command, timeout=45.0):
            self.job_active = False
            self.status(f"PBS job {self.job_id} cancellation confirmed")
        else:
            self.status(
                f"WARNING: could not verify cancellation of PBS job {self.job_id}"
            )

    def stop_remote_relay(self) -> None:
        if not self.relay_started:
            return
        pid_file = str(PurePosixPath(self.remote_run_dir) / "relay.pid")
        command = "\n".join(
            [
                f"pid_file={shlex.quote(pid_file)}",
                "test -r \"$pid_file\" || exit 0",
                "pid=$(cat \"$pid_file\")",
                "case \"$pid\" in *[!0-9]*|'') exit 0;; esac",
                "kill -0 \"$pid\" 2>/dev/null || exit 0",
                "args=$(ps -p \"$pid\" -o args=)",
                f"case \"$args\" in *{shlex.quote(self.remote_relay)}*) kill \"$pid\";; esac",
            ]
        )
        if self.cleanup_remote(command, timeout=30.0):
            self.relay_started = False
        else:
            self.status("WARNING: could not verify relay shutdown on login01")

    def close_master(self) -> None:
        if self.master is None:
            return
        try:
            if self.master.poll() is None and self.control_socket.exists():
                subprocess.run(
                    [
                        "ssh",
                        "-S",
                        str(self.control_socket),
                        "-O",
                        "exit",
                        self.args.ssh_target,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            self.master.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                self.master.terminate()
                self.master.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.master.kill()
                    self.master.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self.master = None

    def stop_tunnel(self) -> None:
        if not self.tunnel_active or not self.control_socket.exists():
            self.tunnel_active = False
            return
        try:
            subprocess.run(
                [
                    "ssh",
                    "-S",
                    str(self.control_socket),
                    "-O",
                    "cancel",
                    "-L",
                    self.forwarding_spec(),
                    self.args.ssh_target,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self.tunnel_active = False

    def cleanup(self, *, interrupted: bool) -> None:
        if self.cleanup_started:
            return
        self.cleanup_started = True
        previous_handlers: dict[int, object] = {}
        cleanup_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            cleanup_signals.append(signal.SIGHUP)
        for cleanup_signal in cleanup_signals:
            previous_handlers[cleanup_signal] = signal.getsignal(cleanup_signal)
            signal.signal(cleanup_signal, signal.SIG_IGN)

        try:
            # Stop local reconnect loops first so cleanup is quiet and bounded.
            self.stop_process(self.client, process_group=True)
            if self.stream_state_started:
                self.publish_stream_state("ended")
                self.status("Metal viewer is replaying the received frames")
            if self.args.close_viewer_on_exit:
                self.stop_process(self.viewer, process_group=True)
            reason = "interruption" if interrupted else "launcher failure"
            self.cancel_remote_job(reason)
            self.stop_remote_relay()
            self.stop_tunnel()
        finally:
            try:
                self.close_master()
            finally:
                if self.wake_guard is not None:
                    self.stop_process(self.wake_guard)
                shutil.rmtree(self.runtime_dir, ignore_errors=True)
                for cleanup_signal, previous_handler in previous_handlers.items():
                    signal.signal(cleanup_signal, previous_handler)

    def run(self) -> None:
        self.check_local_prerequisites()
        self.start_wake_guard()
        self.build_viewer()
        self.start_master()
        self.preflight_cluster()
        self.start_relay()
        self.start_tunnel()
        self.start_client()
        self.start_viewer()
        self.submit_job()
        self.monitor_job()
        self.show_job_tail()
        self.require_successful_simulation()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.post_job_wait < 0:
        parser().error("--post-job-wait must not be negative")
    launcher = ClusterStreamLauncher(args)
    if args.dry_run:
        launcher.print_dry_run()
        shutil.rmtree(launcher.runtime_dir, ignore_errors=True)
        return 0
    interrupted = False
    try:
        launcher.run()
        launcher.status("simulation and streaming run finished")
        return 0
    except KeyboardInterrupt:
        interrupted = True
        print("", file=sys.stderr)
        launcher.status("interrupted")
        return 130
    except (LaunchError, OSError, subprocess.SubprocessError) as error:
        launcher.status(f"ERROR: {error}")
        return 1
    finally:
        launcher.cleanup(interrupted=interrupted)


def handle_termination(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handle_termination)
    raise SystemExit(main())
