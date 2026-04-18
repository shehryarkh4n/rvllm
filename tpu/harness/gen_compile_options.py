#!/usr/bin/env python3
"""Generate compile_options protobuf for PJRT multi-device SPMD compilation.

Produces a serialized xla::CompileOptionsProto that the Rust PJRT client
passes directly to PJRT_Client_Compile. Intended for TPU v6e-4 (4 chips)
with tensor parallelism across all 4 devices.

Usage:
    python3 gen_compile_options.py --num-partitions 4 --output compile_options_tp4.pb
    python3 gen_compile_options.py --num-partitions 1 --output compile_options_tp1.pb
"""
import argparse
import sys


def build_compile_options_proto(num_replicas: int, num_partitions: int) -> bytes:
    """Build a serialized xla::CompileOptionsProto using JAX's xla_client.

    The proto contains:
      - ExecutableBuildOptionsProto with num_replicas, num_partitions,
        use_spmd_partitioning=True, and a DeviceAssignment.
      - DeviceAssignment maps (replica, partition) -> device ordinal.

    For TP=4 on v6e-4: 1 replica x 4 partitions, device assignment
    [[0, 1, 2, 3]].
    """
    from jax._src.lib import xla_client

    opts = xla_client.CompileOptions()
    opts.num_replicas = num_replicas
    opts.num_partitions = num_partitions

    total_devices = num_replicas * num_partitions

    # Build device assignment: shape [num_replicas, num_partitions]
    # Each entry is the device ordinal. For a single-replica TP setup,
    # replica 0 gets devices 0..num_partitions-1.
    import numpy as np
    device_assignment = np.arange(total_devices, dtype=np.int64).reshape(
        num_replicas, num_partitions
    )
    opts.device_assignment = xla_client.DeviceAssignment.create(device_assignment)

    # Enable SPMD partitioning (required for sharded HLO)
    opts.executable_build_options.use_spmd_partitioning = True

    # Serialize to protobuf bytes -- this is the format PJRT_Client_Compile expects
    # xla_client.CompileOptions has a SerializeAsString method that returns
    # the serialized xla::CompileOptionsProto
    try:
        serialized = opts.SerializeAsString()
    except AttributeError:
        # Older JAX versions: build the proto manually
        serialized = _build_proto_manual(num_replicas, num_partitions)

    return serialized


def _build_proto_manual(num_replicas: int, num_partitions: int) -> bytes:
    """Fallback: hand-encode the CompileOptionsProto as raw protobuf bytes.

    CompileOptionsProto layout (from xla/pjrt/compile_options.proto):
      field 4: ExecutableBuildOptionsProto executable_build_options
        field 1: int64 num_replicas
        field 2: int64 num_partitions
        field 3: bool use_spmd_partitioning  (= true)
        field 6: DeviceAssignmentProto device_assignment
          field 1: int32 replica_count
          field 2: int32 computation_count  (= num_partitions)
          field 3: repeated ComputationDevice computation_devices
            field 1: repeated int32 replica_device_ids

    Wire format: each field is (field_number << 3 | wire_type).
    varint=0, length-delimited=2.
    """
    def varint(val):
        out = bytearray()
        while val > 0x7F:
            out.append((val & 0x7F) | 0x80)
            val >>= 7
        out.append(val & 0x7F)
        return bytes(out)

    def field_varint(field_num, val):
        tag = varint((field_num << 3) | 0)  # wire type 0 = varint
        return tag + varint(val)

    def field_bytes(field_num, data):
        tag = varint((field_num << 3) | 2)  # wire type 2 = length-delimited
        return tag + varint(len(data)) + data

    # Build DeviceAssignment.ComputationDevice for each partition
    # computation_devices[p].replica_device_ids = [device_ordinal_for_replica_0]
    comp_devices = b""
    for p in range(num_partitions):
        # For each partition, list replica device IDs
        # With 1 replica: just the device ordinal = p
        device_ids = b""
        for r in range(num_replicas):
            device_ids += field_varint(1, r * num_partitions + p)
        comp_devices += field_bytes(3, device_ids)

    # DeviceAssignmentProto
    device_assignment = (
        field_varint(1, num_replicas) +       # replica_count
        field_varint(2, num_partitions) +      # computation_count
        comp_devices                            # computation_devices
    )

    # ExecutableBuildOptionsProto
    exec_build_opts = (
        field_varint(1, num_replicas) +        # num_replicas
        field_varint(2, num_partitions) +       # num_partitions
        field_varint(3, 1) +                   # use_spmd_partitioning = true
        field_bytes(6, device_assignment)       # device_assignment
    )

    # CompileOptionsProto: field 4 = executable_build_options
    compile_options = field_bytes(4, exec_build_opts)

    return compile_options


def decode_summary(data: bytes) -> str:
    """Produce a human-readable summary of the compile options proto."""
    lines = []
    lines.append(f"raw bytes ({len(data)}): {data.hex()}")

    try:
        from jax._src.lib import xla_client
        opts = xla_client.CompileOptions.ParseFromString(data)
        lines.append(f"num_replicas:          {opts.num_replicas}")
        lines.append(f"num_partitions:        {opts.num_partitions}")
        lines.append(f"use_spmd_partitioning: {opts.executable_build_options.use_spmd_partitioning}")
        da = opts.device_assignment
        if da is not None:
            lines.append(f"device_assignment:     {da}")
    except Exception:
        # Manual decode from raw bytes
        lines.append("(JAX not available for full decode, showing raw hex)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate compile_options.pb for PJRT SPMD compilation"
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1,
        help="Number of replicas (default: 1)"
    )
    parser.add_argument(
        "--num-partitions", type=int, default=4,
        help="Number of partitions / TP degree (default: 4)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="compile_options_tp4.pb",
        help="Output file path (default: compile_options_tp4.pb)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print human-readable summary of the generated proto"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Force manual protobuf encoding (skip JAX dependency)"
    )
    args = parser.parse_args()

    if args.num_partitions < 1:
        print("ERROR: --num-partitions must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.num_replicas < 1:
        print("ERROR: --num-replicas must be >= 1", file=sys.stderr)
        sys.exit(1)

    total = args.num_replicas * args.num_partitions
    print(f"generating CompileOptionsProto: {args.num_replicas} replica(s) x "
          f"{args.num_partitions} partition(s) = {total} device(s)", file=sys.stderr)

    if args.manual:
        data = _build_proto_manual(args.num_replicas, args.num_partitions)
    else:
        try:
            data = build_compile_options_proto(args.num_replicas, args.num_partitions)
        except ImportError:
            print("JAX not available, using manual protobuf encoding", file=sys.stderr)
            data = _build_proto_manual(args.num_replicas, args.num_partitions)

    with open(args.output, "wb") as f:
        f.write(data)
    print(f"wrote {args.output} ({len(data)} bytes)", file=sys.stderr)

    if args.summary or True:  # always print summary
        print(file=sys.stderr)
        print("--- compile options summary ---", file=sys.stderr)
        print(f"num_replicas:          {args.num_replicas}", file=sys.stderr)
        print(f"num_partitions:        {args.num_partitions}", file=sys.stderr)
        print(f"use_spmd_partitioning: true", file=sys.stderr)
        print(f"device_assignment:     [{args.num_replicas}x{args.num_partitions}] "
              f"ordinals 0..{total-1}", file=sys.stderr)
        print(f"total devices:         {total}", file=sys.stderr)
        print(f"proto hex:             {data.hex()}", file=sys.stderr)
        print(f"proto bytes:           {list(data)}", file=sys.stderr)


if __name__ == "__main__":
    main()
