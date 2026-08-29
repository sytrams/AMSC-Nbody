# Live cluster-to-Mac streaming tutorial

This is the firewall-safe workflow verified for this cluster. The compute node
connects **outward** to an authenticated relay on the login node. The Mac
connects to the relay through SSH.

The data path is:

```text
GPU simulator
  -> automatically started nbody_stream_server
  -> authenticated outbound connection to login01:4748
  -> nbody_stream_relay on login01
  -> SSH tunnel
  -> nbody_stream_client on the Mac
  -> nbody_viewer using Metal on the Mac
```

If the relay, SSH tunnel, or Mac client is unavailable, `.nbsnap` files stay in
the cluster spool. A file is removed only after the Mac validates and
acknowledges it.

The concrete paths used below are:

```text
cluster login:   u10774182@10.78.18.100
cluster project: /home/u10774182/AMSC-Nbody-fresh
local project:   /Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody
relay source:    login01:4748
relay client:    login01 loopback port 4747
Mac endpoint:    127.0.0.1:4747
```

## Part 1: one-time cluster build

### 1. Connect to the login node

Run on the Mac:

```bash
ssh u10774182@10.78.18.100
```

### 2. Request a GPU shell

Run on the login node:

```bash
qsub -I -q gpu
```

Wait until the prompt moves to a GPU node.

### 3. Enter the CUDA container

Use the CUDA development image already used for this project. The following
image was verified on this cluster:

```bash
export NBODY_CONTAINER=/home/u10774182/AMSC-Nbody-ci/releases/a1ae8c658170308a75c636631a57893a67b2d08a/nbody-tests.sif
cd /home/u10774182/AMSC-Nbody-fresh
apptainer shell --nv "$NBODY_CONTAINER"
```

Do not run `module load` after the prompt changes to `Apptainer>`.

### 4. Configure an isolated build

Run inside Apptainer:

```bash
cd /home/u10774182/AMSC-Nbody-fresh
cmake -S . -B build/cluster-relay -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=ON \
  -DNBODY_BUILD_SIMULATION=ON \
  -DNBODY_BUILD_STREAMING=ON \
  -DNBODY_BUILD_VIEWER=OFF
```

### 5. Compile all three cluster programs

```bash
cmake --build build/cluster-relay \
  --target nbody nbody_stream_server nbody_stream_relay \
  --parallel
```

Verify them:

```bash
./build/cluster-relay/nbody --help
./build/cluster-relay/nbody_stream_server --help
./build/cluster-relay/nbody_stream_relay --help
```

The help for `nbody` must include `--stream-relay-host`. The server help must
include `--relay-host`.

Exit Apptainer and the temporary GPU shell when the build is finished:

```bash
exit
exit
```

## Part 2: one-time authentication token

Reconnect to the login node if necessary:

```bash
ssh u10774182@10.78.18.100
cd /home/u10774182/AMSC-Nbody-fresh
```

Create one private 256-bit token on the shared home filesystem:

```bash
./scripts/create-stream-token.sh "$PWD/.nbody-stream-token"
ls -l "$PWD/.nbody-stream-token"
```

The permissions must be `-rw-------`. Never paste this token into commands,
copy it to the Mac, or commit it to Git. Both the login and compute nodes read
the same file from shared storage.

This step is performed only once. Future runs reuse the file.

## Part 3A: one-command live streaming run

The recommended launcher replaces the four-terminal procedure with one command
on the Mac. It opens one SSH master connection, so password or MFA
authentication is performed once and reused. It then starts the relay and SSH
tunnel, starts the Mac client and Metal viewer, submits a PBS GPU batch job,
monitors it, and stops the local processes, temporary relay, and tunnel when
the job finishes. The viewer is intentionally left open after network cleanup.

Run locally on the Mac:

```bash
cd /Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody
./scripts/run-cluster-stream.py \
  --input data/two_body.bin \
  --steps 1000000 \
  --stream-max-particles 2
```

Authenticate at the SSH prompt if asked. No other terminal is required. The
viewer window opens automatically and begins updating as complete frames
arrive. After the stream ends, it loops the current run from frame zero. A
later launch with the same `--output-dir` reuses that viewer and resumes live
display rather than opening another window.

The launcher also runs `caffeinate -i` for its lifetime on macOS. This keeps
the Mac itself awake so the VPN, SSH tunnel, and downloads continue, while the
display remains free to turn off according to the normal display-sleep setting.

Frames are also retained in:

```text
/Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody/received-frames
```

The durable cluster spool is
`/home/u10774182/AMSC-Nbody-fresh/nbody-frames-live`. Per-run relay and PBS logs
are retained below `.nbody-stream-runs` in the cluster project.

In the viewer, drag to rotate, use two-finger scrolling or the arrow keys to
pan, and pinch or use `+`/`-` to zoom. Click a particle to follow its sampled
slot across frames, press Escape to release it, `F` to fit the current frame,
or `R` to reset rotation. Use `--no-viewer` to keep downloading without opening
a window.
Use `--viewer-replay-fps FPS` to set loop speed, or
`--close-viewer-on-exit` if the window should close with the launcher.

For a real dataset, change the input and particle limit:

```bash
./scripts/run-cluster-stream.py \
  --input data/YOUR_FILE.bin \
  --steps 100000000 \
  --stream-max-particles 100000
```

Use `--help` to see all parameters. `Ctrl+C` performs an orderly shutdown and
cancels an active PBS job by default. Local reconnect loops are stopped first;
the launcher then verifies `qdel`, force-deletes a job that does not exit, and
reconnects to `login01` for cleanup if its multiplexed SSH connection has died.
Add `--leave-job-running` if the GPU job should continue producing durable
spool files after the Mac disconnects.

Frame sampling is checked after each completed simulation step. If a large
dataset shows frame zero but does not advance, run with `--profile-stages` to
report whether spatial-tree construction or Barnes-Hut acceleration is taking
the time; `--sample-rate` limits transfer frequency but cannot interrupt a GPU
step to create intermediate physical states.
At normal completion, the compute-side server stays alive for up to 30 seconds
to drain and acknowledge the remaining frames. Use
`--stream-drain-seconds SECONDS` to adjust this bounded wait; an undrained
backlog remains in the durable cluster spool.

To inspect the exact PBS job without connecting or submitting it:

```bash
./scripts/run-cluster-stream.py --dry-run
```

## Part 3B: manual four-terminal live streaming run

Use four terminals and start them in the order below.

### Terminal 1 — start the relay on the login node

On the Mac, open a terminal and connect:

```bash
ssh u10774182@10.78.18.100
cd /home/u10774182/AMSC-Nbody-fresh
```

Start the already-built relay:

```bash
NBODY_SKIP_BUILD=1 \
NBODY_RELAY_EXECUTABLE="$PWD/build/cluster-relay/nbody_stream_relay" \
NBODY_RELAY_TOKEN_FILE="$PWD/.nbody-stream-token" \
NBODY_RELAY_SOURCE_BIND=0.0.0.0 \
NBODY_RELAY_SOURCE_PORT=4748 \
NBODY_RELAY_CLIENT_BIND=127.0.0.1 \
NBODY_RELAY_CLIENT_PORT=4747 \
./scripts/run-stream-relay.sh
```

Expected output:

```text
N-body relay source listener on 0.0.0.0:4748
Client listener on 127.0.0.1:4747
```

Leave this terminal running. Port 4748 accepts compute sources only after they
prove possession of the token. Port 4747 is loopback-only and is not exposed
to the cluster network.

### Terminal 2 — open the SSH tunnel on the Mac

Run locally on the Mac:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:4747:127.0.0.1:4747 \
  u10774182@10.78.18.100
```

A successful tunnel normally prints nothing. Leave it running.

### Terminal 3 — start the client on the Mac

Run locally:

```bash
cd /Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody
NBODY_STREAM_HOST=127.0.0.1 \
NBODY_STREAM_PORT=4747 \
NBODY_FRAME_OUTPUT_DIR="$PWD/received-frames" \
./scripts/run-stream-client.sh
```

Leave the client running. It may reconnect until the GPU simulation starts.

### Terminal 4 — start the GPU simulation

Connect and request a GPU:

```bash
ssh u10774182@10.78.18.100
qsub -I -q gpu
```

Enter the same CUDA container:

```bash
export NBODY_CONTAINER=/home/u10774182/AMSC-Nbody-ci/releases/a1ae8c658170308a75c636631a57893a67b2d08a/nbody-tests.sif
cd /home/u10774182/AMSC-Nbody-fresh
apptainer shell --nv "$NBODY_CONTAINER"
```

Prepare the durable spool:

```bash
cd /home/u10774182/AMSC-Nbody-fresh
export NBODY_SPOOL="$PWD/nbody-frames-live"
mkdir -p "$NBODY_SPOOL"
```

Run a short two-particle verification first:

```bash
./build/cluster-relay/nbody \
  --input data/two_body.bin \
  --time-step 0.001 \
  --steps 10000 \
  --theta 0.5 \
  --softening 1e-6 \
  --cluster \
  --stream-dir "$NBODY_SPOOL" \
  --sample-rate 60 \
  --stream-max-particles 2 \
  --stream-relay-host login01 \
  --stream-relay-port 4748 \
  --stream-relay-token-file "$PWD/.nbody-stream-token" \
  --stream-reconnect-ms 1000
```

The simulator starts `nbody_stream_server` automatically. Expected messages
across the terminals include:

```text
Started N-body stream server
Authenticated with relay; waiting for a client
Relay paired a client
Received run-...frame-....nbsnap
```

After this succeeds, use the real dataset, for example:

```bash
./build/cluster-relay/nbody \
  --input data/test_spiral.bin \
  --time-step 0.001 \
  --steps 10000 \
  --theta 0.5 \
  --softening 1e-6 \
  --cluster \
  --stream-dir "$NBODY_SPOOL" \
  --sample-rate 60 \
  --stream-max-particles 100000 \
  --stream-relay-host login01 \
  --stream-relay-port 4748 \
  --stream-relay-token-file "$PWD/.nbody-stream-token"
```

## Part 4: disconnect and reconnect behavior

You may stop the Mac client or SSH tunnel at any time. The GPU simulation keeps
running, the source reconnects automatically, and unacknowledged frames remain
in `NBODY_SPOOL`.

To reconnect:

1. restart the relay if it stopped;
2. reopen the SSH tunnel;
3. rerun `run-stream-client.sh`.

The backlog is delivered before new live frames. To retrieve only the backlog
and exit, add `--once` to the client command:

```bash
./scripts/run-stream-client.sh --once
```

## Part 5: retrieve a backlog after the simulation ended

The automatic source exits with the simulator. Because the spool is in the
shared project directory, a standalone source can run on the login node.

Keep the relay and SSH tunnel running, then open another login-node terminal:

```bash
ssh u10774182@10.78.18.100
cd /home/u10774182/AMSC-Nbody-fresh
export NBODY_SPOOL="$PWD/nbody-frames-live"
```

Start the stored-backlog source:

```bash
./build/cluster-relay/nbody_stream_server \
  --spool-dir "$NBODY_SPOOL" \
  --relay-host 127.0.0.1 \
  --relay-port 4748 \
  --relay-token-file "$PWD/.nbody-stream-token" \
  --reconnect-ms 1000
```

On the Mac, use either follow mode or `--once`. Stop the standalone source with
`Ctrl+C` after the backlog is empty.

## Normal shutdown order

1. Stop or allow the simulation to finish.
2. Stop the Mac client with `Ctrl+C`.
3. Stop the SSH tunnel with `Ctrl+C`.
4. Stop the login-node relay with `Ctrl+C`.

Unacknowledged frames remain recoverable in the spool.

## Troubleshooting

### `Relay token file must not be accessible by group or other users`

Run on the cluster:

```bash
chmod 600 /home/u10774182/AMSC-Nbody-fresh/.nbody-stream-token
```

### `Relay connection interrupted: Could not connect`

Confirm Terminal 1 is running and says it is listening on source port 4748:

```bash
ss -ltn | grep 4748
```

### Client says `Connection refused`

Confirm the SSH tunnel is still running and the relay client listener is
`127.0.0.1:4747`. Do not tunnel to the GPU node; the cluster firewall blocks
that direction.

### `Could not open particle input file`

Use literal underscores and verify the path:

```bash
ls -lh data/test_spiral.bin
```

Do not write `test\_spiral.bin` in a shell command.

### `Address already in use`

Only one relay should use ports 4747 and 4748. Stop the older relay or choose
new source/client ports consistently in the relay, simulation, and SSH tunnel.
