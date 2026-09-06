#!/bin/bash
# Request A100 quota increase across all supported regions
# Project: radiant-micron-464720-r6

PROJECT_ID="radiant-micron-464720-r6"

# A2 regions that support a2-highgpu-1g
REGIONS=(
  "us-central1"
  "us-east1"
  "us-west1"
  "us-west3"
  "us-west4"
  "europe-west4"
  "me-west1"
  "asia-northeast1"
  "asia-northeast3"
  "asia-southeast1"
)

# The quota metric for A100 GPUs
# For a2-highgpu-1g, you need "NVIDIA_A100_GPUS" quota
QUOTA_METRIC="compute.googleapis.com/nvidia_a100_gpus"

# Requested quota (1 GPU is enough for Gemma-2-9B)
REQUESTED_QUOTA=1

echo "=========================================="
echo " A100 Quota Request Script"
echo " Project: $PROJECT_ID"
echo "=========================================="
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
  echo "ERROR: Not authenticated. Run 'gcloud auth login' first."
  exit 1
fi

# Set the project
gcloud config set project $PROJECT_ID

echo "Requesting quota increase for NVIDIA_A100_GPUS in all regions..."
echo ""

for REGION in "${REGIONS[@]}"; do
  echo "----------------------------------------"
  echo "Region: $REGION"
  
  # Check current quota
  CURRENT=$(gcloud compute regions describe $REGION \
    --project=$PROJECT_ID \
    --format="value(quotas[metric=NVIDIA_A100_GPUS].limit)" 2>/dev/null)
  
  if [ -z "$CURRENT" ]; then
    CURRENT=0
  fi
  
  echo "  Current quota: $CURRENT"
  
  if [ "$CURRENT" -ge "$REQUESTED_QUOTA" ]; then
    echo "  ✓ Already have sufficient quota"
    continue
  fi
  
  # Request quota increase using Cloud Quotas API
  # Note: This creates a quota increase request that needs approval
  echo "  Requesting increase to $REQUESTED_QUOTA..."
  
  # Method 1: Using gcloud alpha quotas (if available)
  if gcloud alpha quotas 2>/dev/null; then
    gcloud alpha quotas quota-preferences create \
      --project=$PROJECT_ID \
      --service=compute.googleapis.com \
      --quota-id="NVIDIA_A100_GPUS-per-project-region" \
      --preferred-value=$REQUESTED_QUOTA \
      --dimensions=region=$REGION \
      --justification="Running ML benchmarks for research paper - need to validate activation probes on Gemma-2-9B model" \
      2>/dev/null && echo "  ✓ Request submitted" || echo "  ✗ Failed (may need manual request)"
  else
    echo "  ✗ gcloud alpha quotas not available"
    echo "  → Manual request needed"
  fi
done

echo ""
echo "=========================================="
echo " Alternative: Request via Console"
echo "=========================================="
echo ""
echo "If programmatic requests failed, use the Cloud Console:"
echo ""
for REGION in "${REGIONS[@]}"; do
  echo "https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID&metric=compute.googleapis.com%2Fnvidia_a100_gpus&location=$REGION"
done
echo ""
echo "Or use the unified quota request page:"
echo "https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID&metric=compute.googleapis.com%2Fnvidia_a100_gpus"
echo ""
echo "Tip: Request quota in ALL regions simultaneously to maximize approval chances."
echo "     Use justification: 'ML research benchmarks - validating activation probes on 9B parameter model'"
