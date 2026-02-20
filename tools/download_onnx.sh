#!/bin/bash
set -e

# Default values
VERSION="1.21.0"
DEVICE="cpu"
OUTPUT_DIR="libs/onnx"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download ONNX Runtime for Linux x64."
    echo ""
    echo "Options:"
    echo "  -v <version>  ONNX Runtime version (default: $VERSION)"
    echo "  -d <device>   Device type (cpu/gpu, default: $DEVICE)"
    echo "  -o <dir>      Output directory (default: $OUTPUT_DIR)"
    echo "  -h            Show this help message"
    echo ""
    exit 0
}

# Parse arguments
while getopts "v:d:o:h" opt; do
    case "$opt" in
        v) VERSION=$OPTARG ;;
        d) DEVICE=$OPTARG ;;
        o) OUTPUT_DIR=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Architecture is fixed to x64 for this script
ARCH="x64"
OS="linux"

# Construct filename and URL
# CPU: onnxruntime-linux-x64-1.21.0.tgz
# GPU: onnxruntime-linux-x64-gpu-1.21.0.tgz
if [ "$DEVICE" == "gpu" ]; then
    FILENAME="onnxruntime-${OS}-${ARCH}-gpu-${VERSION}"
else
    FILENAME="onnxruntime-${OS}-${ARCH}-${VERSION}"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${FILENAME}.tgz"

echo ">>> Downloading ONNX Runtime v${VERSION} (${DEVICE}) for ${OS}-${ARCH}..."
echo ">>> URL: ${URL}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Download
TEMP_TGZ="${FILENAME}.tgz"
curl -L -o "$TEMP_TGZ" "$URL"

# Extract
echo ">>> Extracting..."
tar -xzf "$TEMP_TGZ"

# Cleanup
rm "$TEMP_TGZ"

echo ">>> Download complete: $OUTPUT_DIR/$FILENAME"
echo ">>> To use this with CMake, set -DONNXRUNTIME_ROOT_DIR=$(pwd)/$FILENAME"
