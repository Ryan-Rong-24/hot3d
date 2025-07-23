#!/bin/bash

# Script to delete all .vrs files in /dataset directory
# Usage: ./delete_vrs_files.sh [options]

set -e  # Exit on error

# Default values
DATASET_PATH="dataset"
DRY_RUN=false
SKIP_CONFIRMATION=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Delete all .vrs files in /dataset directory"
    echo ""
    echo "Options:"
    echo "  -p, --path PATH     Dataset path (default: /dataset)"
    echo "  -d, --dry-run       Show files that would be deleted without deleting"
    echo "  -y, --yes           Skip confirmation prompt"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Delete .vrs files in /dataset (with confirmation)"
    echo "  $0 --dry-run        # Show what would be deleted"
    echo "  $0 --yes            # Delete without confirmation"
    echo "  $0 -p /other/path   # Delete .vrs files in /other/path"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRMATION=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if dataset path exists
if [[ ! -d "$DATASET_PATH" ]]; then
    echo "Error: Dataset path '$DATASET_PATH' does not exist"
    exit 1
fi

echo "Searching for .vrs files in: $DATASET_PATH"

# Find all .vrs files recursively
mapfile -t vrs_files < <(find "$DATASET_PATH" -name "*.vrs" -type f)

# Check if any .vrs files were found
if [[ ${#vrs_files[@]} -eq 0 ]]; then
    echo "No .vrs files found"
    exit 0
fi

echo "Found ${#vrs_files[@]} .vrs files:"
for file in "${vrs_files[@]}"; do
    echo "  $file"
done

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "DRY RUN - Files that would be deleted:"
    for file in "${vrs_files[@]}"; do
        echo "  $file"
    done
    exit 0
fi

# Confirmation prompt
if [[ "$SKIP_CONFIRMATION" == false ]]; then
    echo ""
    read -p "Are you sure you want to delete ${#vrs_files[@]} .vrs files? (yes/no): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        echo "Operation cancelled"
        exit 0
    fi
fi

# Delete files
echo ""
echo "Deleting files..."
deleted_count=0
failed_count=0

for file in "${vrs_files[@]}"; do
    if rm "$file" 2>/dev/null; then
        echo "Deleted: $file"
        deleted_count=$((deleted_count + 1))
    else
        echo "Failed to delete: $file"
        failed_count=$((failed_count + 1))
    fi
done

echo ""
echo "Summary: $deleted_count files deleted, $failed_count failures" 