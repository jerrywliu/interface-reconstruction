#!/bin/bash
########################################################
# Run square experiments with getArcFacet logging enabled
#
# This is a convenience wrapper around the Python script.
# For more options, use the Python script directly:
#   python -m test.geoms.getarcfacet.run_squares_with_logging --help
#
# Usage:
#   ./run_squares_with_logging.sh [log_file] [resolution]
#
# Examples:
#   ./run_squares_with_logging.sh
#   ./run_squares_with_logging.sh my_log.log 0.50
#   ./run_squares_with_logging.sh my_log.log 1.50
#
# Default log file: get_arc_facet_calls.log
# Default resolution: from config (typically 0.5)
########################################################

LOG_FILE="${1:-get_arc_facet_calls.log}"
RESOLUTION="${2:-}"

# Build the command
CMD="python -m test.geoms.getarcfacet.run_squares_with_logging --log-file $LOG_FILE"

if [ -n "$RESOLUTION" ]; then
    CMD="$CMD --resolution $RESOLUTION"
fi

echo "=========================================="
echo "Running square experiments with logging"
echo "=========================================="
echo ""
echo "Command: $CMD"
echo ""

# Execute the Python script
$CMD

