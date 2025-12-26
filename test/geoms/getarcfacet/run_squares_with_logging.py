#!/usr/bin/env python3
"""
Run square experiments with getArcFacet logging enabled.

This script enables logging of all getArcFacet calls before running
the square experiments, allowing you to capture challenging test cases
derived from `experiments/static/squares.py`.

Usage:
    python -m test.geoms.getarcfacet.run_squares_with_logging [OPTIONS]

Examples:
    # Run with default settings (resolution from config)
    python -m test.geoms.getarcfacet.run_squares_with_logging

    # Run with fine resolution (more challenging)
    python -m test.geoms.getarcfacet.run_squares_with_logging --resolution 0.50

    # Run with coarse resolution (faster, but easier cases)
    python -m test.geoms.getarcfacet.run_squares_with_logging --resolution 1.50

    # Run with custom log file and specific algorithm
    python -m test.geoms.getarcfacet.run_squares_with_logging --log-file my_log.log --facet-algo circular --resolution 0.64
"""

import argparse

from util.config import read_yaml
from util.logging.get_arc_facet_logger import enable_logging, disable_logging, get_stats

DEFAULT_CONFIG = "static/square"


def log_mesh_settings(config_setting: str, override_resolution):
    """Emit diagnostic logs about the mesh configuration being used."""
    config = read_yaml(f"config/{config_setting}.yaml")
    mesh_cfg = config.get("MESH", {})
    grid_size = mesh_cfg.get("GRID_SIZE", "unknown")
    default_resolution = mesh_cfg.get("RESOLUTION", "unknown")

    print(f"Config setting: {config_setting}")
    print(f"  Mesh grid size: {grid_size}")
    print(f"  Default resolution from config: {default_resolution}")

    if override_resolution is None:
        print("  Using default resolution from config (override not provided).")
        effective_resolution = default_resolution
    else:
        print(f"  Overriding resolution to: {override_resolution}")
        effective_resolution = override_resolution

    return effective_resolution


def main():
    parser = argparse.ArgumentParser(
        description="Run square experiments with getArcFacet logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution options:
  Finer resolutions (more challenging, slower):
    --resolution 0.50  (fine)
    --resolution 0.64  (medium-fine)
  
  Coarser resolutions (easier, faster):
    --resolution 1.00  (medium-coarse)
    --resolution 1.28  (coarse)
    --resolution 1.50  (very coarse, easiest)
        """,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="get_arc_facet_calls.log",
        help="Path to log file (default: get_arc_facet_calls.log)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Configuration setting for square experiments (default: static/square)",
    )
    parser.add_argument(
        "--facet-algo",
        type=str,
        choices=["circular", "safe_circle"],
        default="circular",
        help="Facet algorithm to use (default: circular)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Grid resolution (default: from config file). "
        "Smaller values = finer grid = more challenging cases. "
        "Options: 0.50, 0.64, 1.00, 1.28, 1.50",
    )
    parser.add_argument(
        "--num-squares",
        type=int,
        default=25,
        help="Number of squares to test (default: 25)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep across all resolutions instead of single experiment",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SQUARE EXPERIMENTS WITH GETARCFACET LOGGING")
    print("=" * 80)
    print(f"Log file: {args.log_file}")
    print(f"Facet algorithm: {args.facet_algo}")
    print(f"Number of squares: {args.num_squares}")
    print(
        f"Mode: {'Parameter sweep (all resolutions)' if args.sweep else 'Single experiment'}"
    )
    print("=" * 80)

    effective_resolution = log_mesh_settings(args.config, args.resolution)
    print(
        f"Effective resolution for this run (after overrides): {effective_resolution}"
    )
    print("=" * 80)
    print()

    # Enable logging BEFORE importing experiment modules
    print("Enabling getArcFacet logging...")
    enable_logging(log_file=args.log_file)
    print()

    try:
        # Now import and run experiments
        # The monkey patch is already in place, so all getArcFacet calls will be logged
        from experiments.static import squares

        if args.sweep:
            print("Running square parameter sweep across resolutions...")
            squares.run_parameter_sweep(args.config, num_squares=args.num_squares)
        else:
            print("Running single square experiment...")
            squares.main(
                config_setting=args.config,
                facet_algo=args.facet_algo,
                resolution=args.resolution,
                num_squares=args.num_squares,
            )

        print()
        print("=" * 80)
        print("EXPERIMENTS COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nError during experiments: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always disable logging when done
        print("\nDisabling logging...")
        disable_logging()

        # Print statistics
        print()
        print("=" * 80)
        print("LOGGING STATISTICS")
        print("=" * 80)
        stats = get_stats()
        for key, value in stats.items():
            if key != "log_file":
                print(f"  {key}: {value}")
        print()
        print(f"Log file: {stats.get('log_file', args.log_file)}")
        print()
        print("To analyze the log file:")
        print(f"  python -m util.logging.analyze_log {args.log_file}")
        print()
        print("To extract failed cases:")
        print(
            f"  python -m util.logging.analyze_log {args.log_file} --extract-failed --output failed_cases.py"
        )
        print()
        print("To extract slow cases (>1s):")
        print(
            f"  python -m util.logging.analyze_log {args.log_file} --extract-slow --threshold 1.0 --output slow_cases.py"
        )
        print()


if __name__ == "__main__":
    main()
